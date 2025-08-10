import modal

# Image that installs all dependencies listed in requirements.txt.
image = modal.Image.debian_slim().pip_install_from_requirements("requirements.txt")

app = modal.App("hello-min", image=image)

@app.function()
def hello(name: str = "world"):
    return f"hello, {name}!"

@app.local_entrypoint()
def main():
    print(hello.remote())  # runs in the cloud
