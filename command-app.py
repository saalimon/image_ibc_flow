import click
import subprocess

@click.group()
def cli():
    """Command Base App to Run Different Scripts"""
    pass

@cli.command()
def predict():
    """Run Prediction Script"""
    try:
        subprocess.run(["python", "./model/ocr_model/predict_number.py"], check=True)
    except subprocess.CalledProcessError as e:
        click.echo(f"Error running predict: {e}")

@cli.command()
def train():
    """Run Training Script"""
    try:
        subprocess.run(["python", "./model/ocr_model/train.py"], check=True)
    except subprocess.CalledProcessError as e:
        click.echo(f"Error running script2: {e}")

@cli.command()
def get_images():
    """Run Get Images Script"""
    try:
        subprocess.run(["python", "./utils/get_images.py"], check=True)
    except subprocess.CalledProcessError as e:
        click.echo(f"Error running script3: {e}")

if __name__ == "__main__":
    cli()