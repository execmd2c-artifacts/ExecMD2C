import numpy as np
import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
	def __init__(self, input_shape):
		super(EncoderBlock, self).__init__()
		self.input_shape = input_shape
		self.width = input_shape[0]
		
		self.conv1 = nn.Sequential(
			nn.Conv2d(self.width, self.width, kernel_size=(3, 1), padding='same',dilation=(1, 1)),  # 3x1 2D Convolution (d=1)
			nn.BatchNorm2d(self.width),
			nn.PReLU(num_parameters=self.width, init=0.3),
			nn.Conv2d(self.width, self.width, kernel_size=(1, 3), padding='same',dilation=(1, 1)),  # 1x3 2D Convolution (d=1)
			nn.BatchNorm2d(self.width),
			nn.PReLU(num_parameters=self.width, init=0.3),

			nn.Conv2d(self.width, self.width, kernel_size=(3, 1), padding='same',dilation=(2, 2)),  # 3x1 2D Convolution (d=2)
			nn.BatchNorm2d(self.width),
			nn.PReLU(num_parameters=self.width, init=0.3),
			nn.Conv2d(self.width, self.width, kernel_size=(1, 3), padding='same',dilation=(2, 2)),  # 1x3 2D Convolution (d=2)
			nn.BatchNorm2d(self.width),
			nn.PReLU(num_parameters=self.width, init=0.3),

			nn.Conv2d(self.width, self.width, kernel_size=(3, 1), padding='same',dilation=(3, 3)),  # 3x1 2D Convolution (d=3)
			nn.BatchNorm2d(self.width),
			nn.PReLU(num_parameters=self.width, init=0.3),
			nn.Conv2d(self.width, self.width, kernel_size=(1, 3), padding='same',dilation=(3, 3)),  # 1x3 2D Convolution (d=3)
			nn.BatchNorm2d(self.width),
			nn.PReLU(num_parameters=self.width, init=0.3),			
		)


		self.conv2 = nn.Sequential(
			nn.Conv2d(self.width,self.width, kernel_size=(3, 3), padding='same'),
			nn.BatchNorm2d(self.width),
			nn.PReLU(num_parameters=self.width, init=0.3)
		)

		self.prelu1 = nn.PReLU(num_parameters=2*self.width, init=0.3)

		self.conv1x1 = nn.Sequential(
			nn.Conv2d(2*self.width,self.width, kernel_size=(1, 1), padding='same'),
			nn.BatchNorm2d(self.width),
			nn.PReLU(num_parameters=self.width, init=0.3)
		)

		self.prelu2 = nn.PReLU(num_parameters=self.width, init=0.3)

		self.Identity = nn.Identity()
	
	def forward(self, x):
		identity = self.Identity(x)
		res1=self.conv1(x)
		res2=self.conv2(x)
		res=self.prelu1(torch.cat((res1,res2),dim=1))
		res=self.conv1x1(res)
		return self.prelu2(identity + res)

class DecoderBlock(nn.Module):
	def __init__(self, input_shape, expansion):
		super(DecoderBlock, self).__init__()
		self.input_shape = input_shape
		self.width = input_shape[0]
		self.expansion_width = self.width*3*expansion

		self.conv1 = nn.Sequential(
			nn.Conv2d(self.width,self.expansion_width, kernel_size=(3,3), padding='same', dilation=2),
			nn.BatchNorm2d(self.expansion_width),
			nn.PReLU(num_parameters=self.expansion_width, init=0.3),
			nn.Conv2d(self.expansion_width,self.expansion_width, kernel_size=(1,3), padding='same', dilation=3),
			nn.BatchNorm2d(self.expansion_width),
			nn.PReLU(num_parameters=self.expansion_width, init=0.3),

			nn.Conv2d(self.expansion_width,self.expansion_width, kernel_size=(3,1), padding='same', dilation=3),
			nn.BatchNorm2d(self.expansion_width),
			nn.PReLU(num_parameters=self.expansion_width, init=0.3),
			
			nn.Conv2d(self.expansion_width,self.width, kernel_size=(3,3), padding='same'),
			nn.BatchNorm2d(self.width),
			nn.PReLU(num_parameters=self.width, init=0.3),
		)

		self.conv2 = nn.Sequential(
			nn.Conv2d(self.width,self.expansion_width, kernel_size=(1,3), padding='same'),
			nn.BatchNorm2d(self.expansion_width),
			nn.PReLU(num_parameters=self.expansion_width, init=0.3),
			
			nn.Conv2d(self.expansion_width,self.expansion_width, kernel_size=(5,1), padding='same'),
			nn.BatchNorm2d(self.expansion_width),
			nn.PReLU(num_parameters=self.expansion_width, init=0.3),

			nn.Conv2d(self.expansion_width,self.expansion_width, kernel_size=(1,5), padding='same'),
			nn.BatchNorm2d(self.expansion_width),
			nn.PReLU(num_parameters=self.expansion_width, init=0.3),

			nn.Conv2d(self.expansion_width,self.width, kernel_size=(3,1), padding='same'),
			nn.BatchNorm2d(self.width),
			nn.PReLU(num_parameters=self.width, init=0.3)
		)

		self.prelu1 = nn.PReLU(num_parameters=2*self.width, init=0.3)

		self.conv1x1 = nn.Sequential(
			nn.Conv2d(2*self.width,self.width, kernel_size=(1,1), padding='same'),
			nn.BatchNorm2d(self.width),
			nn.PReLU(num_parameters=self.width, init=0.3)
		)

		self.prelu2 = nn.PReLU(num_parameters=self.width, init=0.3)

		self.Identity = nn.Identity()
	
	def forward(self, x):
		identity = self.Identity(x)
		res1=self.conv1(x)
		res2=self.conv2(x)
		res=self.prelu1(torch.cat((res1,res2),dim=1))
		res=self.conv1x1(res)
		return self.prelu2(identity + res)	
	

class RecurrentBlock(nn.Module):
	def __init__(self, input_size, hidden_size, keep_dim=False):
		super(RecurrentBlock, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.recurrent_block = nn.GRU(self.input_size, self.hidden_size, batch_first=True)
		self.keep_dim = keep_dim
		if keep_dim:
			self.fc = nn.Linear(self.hidden_size, self.input_size)
	
	def forward(self, x):
		x, hidden = self.recurrent_block(x)
		if self.keep_dim:
			x = self.fc(x)
		return x, hidden

class Encoder(nn.Module):
	def __init__(self, input_shape):
		super(Encoder, self).__init__()
		self.input_shape = input_shape
		self.input_size = np.prod(input_shape)
		self.width = input_shape[0]

		self.encoder = nn.Sequential(
			nn.Conv2d(self.width,self.width, kernel_size=(5,5), padding='same'),
			nn.BatchNorm2d(self.width),
			nn.PReLU(num_parameters=self.width, init=0.3),
			EncoderBlock(self.input_shape),
			nn.Flatten(),
		)
	
	def forward(self, x):
		x = self.encoder(x)
		# x = self.encoder_fc(x)
		return x

class Decoder(nn.Module):
	def __init__(self, input_shape, expansion):
		super(Decoder, self).__init__()
		self.input_shape = input_shape
		self.input_size = np.prod(input_shape)
		self.width = input_shape[0]

		self.decoder = nn.Sequential(
			nn.Conv2d(self.width,self.width,kernel_size=(5,5),padding='same'),
			nn.BatchNorm2d(self.width),
			nn.PReLU(num_parameters=self.width, init=0.3),
			DecoderBlock(self.input_shape, expansion),
			DecoderBlock(self.input_shape, expansion),
			nn.Sigmoid()
		)

	def forward(self, x):
		x = self.decoder(x)
		return x

class Classifier(nn.Module):
	def __init__(self, input_size, num_classes):
		super(Classifier, self).__init__()
		self.num_classes = num_classes
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(input_size, 512),
			nn.ReLU(),
			nn.Linear(512, 128),
			nn.ReLU(),
			nn.Linear(128, self.num_classes),
		)
	
	def forward(self, x):
		x = self.classifier(x)
		return x

if __name__ == "__main__":
    torch.manual_seed(42)
    passed = 0
    failed = 0
    def check(test_name, condition, detail=""):
        global passed, failed
        if bool(condition):
            passed += 1
            print(f"  [{test_name}] PASS")
        else:
            failed += 1
            suffix = f" - {detail}" if detail else ""
            print(f"  [{test_name}] FAIL{suffix}")
    def skip_checks(count, reason):
        global failed
        failed += count
        print(f"  [skipped {count} check(s)] FAIL - {reason}")
    print("RSCNet: 3-group automated test suite")
    for name, module, x, shape in [
        ("EncoderBlock", EncoderBlock((1, 6, 8)), torch.randn(2, 1, 6, 8, requires_grad=True), (2, 1, 6, 8)),
        ("DecoderBlock", DecoderBlock((1, 6, 8), 1), torch.randn(2, 1, 6, 8, requires_grad=True), (2, 1, 6, 8)),
    ]:
        try:
            y = module(x)
            check(f"{name} output not None", y is not None)
            if y is not None:
                check(f"{name} output shape", tuple(y.shape) == shape)
                check(f"{name} output finite", torch.isfinite(y).all().item())
                y.sum().backward()
                check(f"{name} residual gradient", x.grad is not None and x.grad.abs().sum().item() > 0)
            else: skip_checks(3, f"{name} returned None")
        except Exception as exc:
            print(f"  [{name}] ERROR - {exc}"); skip_checks(4, f"{name} exception")
    try:
        module = RecurrentBlock(4, 6, keep_dim=True)
        y, h = module(torch.randn(2, 5, 4, requires_grad=True))
        check("RecurrentBlock output not None", y is not None)
        if y is not None:
            check("RecurrentBlock output shape", tuple(y.shape) == (2, 5, 4))
            check("RecurrentBlock output finite", torch.isfinite(y).all().item())
            y.sum().backward()
            check("RecurrentBlock state gradients", module.recurrent_block.weight_ih_l0.grad is not None and tuple(h.shape) == (1, 2, 6))
        else: skip_checks(3, "RecurrentBlock returned None")
    except Exception as exc:
        print(f"  [RecurrentBlock] ERROR - {exc}"); skip_checks(4, "RecurrentBlock exception")
    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    if failed != 0:
        raise SystemExit(1)