import modal

app = modal.App("hello-min")

@app.function()
def hello(name: str = "world"):
    return f"hello, {name}!"

@app.local_entrypoint()
def main():
    print(hello.remote())  # runs in the cloud
