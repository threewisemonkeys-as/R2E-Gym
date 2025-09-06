import docker

repo = "slimshetty/swebench-verified"
tag = "sweb.eval.x86_64.sympy__sympy-24562"
client = docker.from_env()

client.images.pull(repository=repo, tag=tag)