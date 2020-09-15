import mujoco_py
import torch


def run_function():
	a = torch.tensor([1,2,3]).cuda()
	print(a)


def main():
	run_function()

main()
