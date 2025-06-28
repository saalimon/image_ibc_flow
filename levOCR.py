import os
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Dict, Optional

from PIL import Image

from levutils import TokenLabelConverter
from dataset import AlignCollateTest, RawDataset
from models import LevOCRModel
from levutils import get_args
from levt import utils as utils_levt
from abinet.utils import CharsetMapper


@dataclass
class OCRConfig:
    """Configuration settings for OCR model"""
    MODEL_DIR: str = "levocr_model.pth"
    IMG_HEIGHT: int = 32
    IMG_WIDTH: int = 128
    BATCH_SIZE: int = 16
    MAX_ITER: int = 2
    THRESHOLD: float = 0.5
    RGB_MODE: bool = True
    INPUT_CHANNELS: int = 3


class LevOCRProcessor:
    def __init__(self):
        self.config = OCRConfig()
        self.device = self._setup_device()
        self.opt = self._initialize_options()
        self.model = self._setup_model()
        self.converter = self._setup_converter()

    def _setup_device(self) -> torch.device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cudnn.benchmark = True
        cudnn.deterministic = True
        return device

    def _initialize_options(self):
        opt = get_args(is_train=False)
        opt.eval = True
        opt.imgH = self.config.IMG_HEIGHT
        opt.imgW = self.config.IMG_WIDTH
        opt.batch_size = self.config.BATCH_SIZE
        opt.max_iter = self.config.MAX_ITER
        opt.rgb = self.config.RGB_MODE
        opt.th = self.config.THRESHOLD
        opt.num_gpu = torch.cuda.device_count()
        opt.model_dir = self.config.MODEL_DIR
        opt.saved_model = opt.model_dir

        return opt

    def _setup_converter(self) -> TokenLabelConverter:
        charset = CharsetMapper(self.opt.dataset_charset_path, max_length=self.opt.batch_max_length)
        self.opt.num_class = charset.num_classes
        indices = charset.char_to_label
        src_dict = utils_levt.build_dict(indices)
        converter = TokenLabelConverter(src_dict.indices)
        self.opt.num_class = len(converter.character)
        return converter

    def _setup_model(self) -> torch.nn.Module:
        if self.opt.rgb:
            self.opt.input_channel = self.config.INPUT_CHANNELS

        model = LevOCRModel(self.opt, utils_levt.build_dict(
            CharsetMapper(self.opt.dataset_charset_path,
                          max_length=self.opt.batch_max_length).char_to_label))
        model = torch.nn.DataParallel(model).to(self.device)

        print(f'Loading pretrained model from {self.opt.saved_model}')
        model.load_state_dict(
            torch.load(self.opt.saved_model, map_location=self.device),
            strict=False
        )
        return model

    def _process_vision_output(self, pred_logit: torch.Tensor,
                               batch_size: int) -> List[str]:
        pred_vision = F.log_softmax(pred_logit, dim=-1)
        pred_vision_max = pred_vision.max(2)[1]
        vision_preds_size = torch.IntTensor([pred_logit.size(1)] * batch_size)
        return self.converter.decode(
            pred_vision_max,
            vision_preds_size,
            ignore_spec_char=True
        )

    def get_results_folder(self, image_folder: str) -> Dict:
        demo_data = RawDataset(root=image_folder, opt=self.opt)
        collate_fn = AlignCollateTest(imgH=self.opt.imgH, imgW=self.opt.imgW)
        demo_loader = torch.utils.data.DataLoader(
            demo_data,
            batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=int(self.opt.workers),
            collate_fn=collate_fn,
            pin_memory=True
        )

        self.model.eval()
        results = {}
        with torch.no_grad():
            for image_tensors, image_path_list in demo_loader:
                batch_size = image_tensors.size(0)
                image = image_tensors.to(self.device)
                out = self.model.module.vision(image)
                pred_logit = out['logits']
                vision_preds_str = self._process_vision_output(pred_logit, batch_size)
                results.update(zip(image_path_list, vision_preds_str))
        return results

    def get_result_single_file(self, image_path: str) -> Optional[str]:
        try:
            collate_fn = AlignCollateTest(imgH=self.opt.imgH, imgW=self.opt.imgW)
            if self.opt.rgb:
                img = Image.open(image_path).convert('RGB')
            else:
                img = Image.open(image_path).convert('L')
            image_tensor, _ = collate_fn([(img, image_path)])
            self.model.eval()
            with torch.no_grad():
                image = image_tensor.to(self.device)
                out = self.model.module.vision(image)
                pred_logit = out['logits']
                vision_preds_str = self._process_vision_output(
                    pred_logit,
                    batch_size=1
                )
                return vision_preds_str[0] if vision_preds_str else None
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None


if __name__ == '__main__':
    processor = LevOCRProcessor()
    results = processor.get_results_folder('./demo_imgs')
    print(results)

    result = processor.get_result_single_file('./demo_imgs/2f67a8a4-2024-10-22-1401-OUTSIDE-6_2_crop_13.jpg')
    print(result)