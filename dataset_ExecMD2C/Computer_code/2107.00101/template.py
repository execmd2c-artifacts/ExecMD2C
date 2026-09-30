# ============================================================
# ground_truth.py - latent-execution Core Model Components
# Source: Computer_Code/latent-execution-main
#
# Contains only model architecture components and direct model helpers.
# No training loops, dataset loading, evaluation supervisors, checkpoint I/O,
# command-line wrappers, or remote dataset requirements.
# ============================================================

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import torch.nn.functional as F
import transformers

import numpy as np


# --- [Original file: models/data_utils/data_utils.py] ---
PAD_ID = 0
EOS_ID = 1
GO_ID = 2


def np_to_tensor(inp, output_type, cuda_flag, eval_flag=False):
	if eval_flag:
		with torch.no_grad():
			if output_type == 'float':
				inp_tensor = Variable(torch.FloatTensor(inp))
			elif output_type == 'int':
				inp_tensor = Variable(torch.LongTensor(inp))
			else:
				print('undefined tensor type')
	else:
		if output_type == 'float':
			inp_tensor = Variable(torch.FloatTensor(inp))
		elif output_type == 'int':
			inp_tensor = Variable(torch.LongTensor(inp))
		else:
			print('undefined tensor type')
	if cuda_flag:
		inp_tensor = inp_tensor.cuda()
	return inp_tensor


class data_utils:
	PAD_ID = PAD_ID
	np_to_tensor = staticmethod(np_to_tensor)


# --- [Original file: models/modules/mlp.py] ---
class MLPModel(nn.Module):
	"""
	Multi-layer perception module.
	"""
	def __init__(self, num_layers, input_size, hidden_size, output_size, dropout_rate, cuda_flag, activation=None):
		super(MLPModel, self).__init__()
		self.num_layers = num_layers
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.cuda_flag = cuda_flag
		self.dropout_rate = dropout_rate
		self.model = nn.Sequential(
			nn.Linear(self.input_size, self.hidden_size),
			nn.Dropout(p=self.dropout_rate),
			nn.ReLU())
		for _ in range(self.num_layers):
			self.model = nn.Sequential(
				self.model,
				nn.Linear(self.hidden_size, self.hidden_size),
				nn.Dropout(p=self.dropout_rate),
				nn.ReLU())
		self.model = nn.Sequential(
			self.model,
			nn.Linear(self.hidden_size, self.output_size))
		if activation is not None:
			self.model = nn.Sequential(
				self.model,
				activation
				)

	def forward(self, inputs):
		return self.model(inputs)


class mlp:
	MLPModel = MLPModel


# --- [Original file: models/model.py] ---
class CodeGenerator(nn.Module):
	def __init__(self, args):
		super(CodeGenerator, self).__init__()
		self.cuda_flag = args.cuda
		self.eval_flag = args.eval
		self.tokenizer_name = args.tokenizer_name
		self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer_name)
		self.vocab_size = len(self.tokenizer)
		self.batch_size = args.batch_size
		self.embedding_size = args.embedding_size
		self.LSTM_hidden_size = args.LSTM_hidden_size
		self.MLP_hidden_size = args.MLP_hidden_size
		self.num_LSTM_layers = args.num_LSTM_layers
		self.num_MLP_layers = args.num_MLP_layers
		self.num_attention_layers = args.num_attention_layers
		self.gradient_clip = args.gradient_clip
		self.lr = args.lr
		self.dropout_rate = args.dropout_rate
		self.max_input_len = args.max_input_len
		self.max_decode_len = args.max_decode_len
		self.io_size = args.io_size
		self.exec_period = args.exec_period
		self.dropout = nn.Dropout(p=self.dropout_rate)
		self.ceLoss = nn.CrossEntropyLoss()
		self.mseLoss = nn.MSELoss()
		self.latent_execution = args.latent_execution
		self.no_partial_execution = args.no_partial_execution
		self.operation_predictor = args.operation_predictor
		self.value_range = args.value_range
		self.value_offset = self.value_range * 4
		self.var_offset = -self.value_range + self.value_offset
		self.decoder_self_attention_flag = args.decoder_self_attention
		self.use_properties = args.use_properties

		self.code_embedding = nn.Embedding(self.vocab_size, self.embedding_size)
		if self.decoder_self_attention_flag or self.use_properties:
			self.code_predictor = mlp.MLPModel(self.num_MLP_layers, self.LSTM_hidden_size * 2, self.MLP_hidden_size, self.vocab_size, self.dropout_rate, self.cuda_flag)
		elif self.operation_predictor:
			self.code_predictor = mlp.MLPModel(self.num_MLP_layers, self.LSTM_hidden_size * 6, self.MLP_hidden_size, self.vocab_size, self.dropout_rate, self.cuda_flag)
		else:
			self.code_predictor = mlp.MLPModel(self.num_MLP_layers, self.LSTM_hidden_size * 4, self.MLP_hidden_size, self.vocab_size, self.dropout_rate, self.cuda_flag)
		self.var_embedding = nn.Embedding(self.value_offset * 2 + 1, self.embedding_size)

		self.input_var_encoder = nn.LSTM(input_size=self.embedding_size, hidden_size=self.LSTM_hidden_size, num_layers=self.num_LSTM_layers, dropout=self.dropout_rate, bidirectional=True, batch_first=True)
		self.output_var_encoder = nn.LSTM(input_size=self.embedding_size, hidden_size=self.LSTM_hidden_size, num_layers=self.num_LSTM_layers, dropout=self.dropout_rate, bidirectional=True, batch_first=True)
		self.operation_encoder = nn.LSTM(input_size=self.embedding_size, hidden_size=self.LSTM_hidden_size, num_layers=self.num_LSTM_layers, dropout=self.dropout_rate, bidirectional=True, batch_first=True)

		self.decoder = nn.LSTM(input_size=self.embedding_size, hidden_size=self.LSTM_hidden_size, num_layers=self.num_LSTM_layers, dropout=self.dropout_rate, bidirectional=True, batch_first=True)
		self.prog_executor = nn.LSTM(input_size=self.LSTM_hidden_size * 2, hidden_size=self.LSTM_hidden_size, num_layers=self.num_LSTM_layers, dropout=self.dropout_rate, bidirectional=True, batch_first=True)

		self.prog_executor_attention = nn.Linear(self.LSTM_hidden_size * 2, self.embedding_size)

		self.encoder_o2i_attention_linear = nn.ModuleList([nn.Linear(self.LSTM_hidden_size * 2, self.LSTM_hidden_size * 2) for _ in range(self.num_attention_layers)])
		self.encoder_i2o_attention_linear = nn.ModuleList([nn.Linear(self.LSTM_hidden_size * 2, self.LSTM_hidden_size * 2) for _ in range(self.num_attention_layers)])
		self.decoder_d2i_attention_linear = nn.Linear(self.LSTM_hidden_size * 2, self.LSTM_hidden_size * 2)
		self.decoder_d2o_attention_linear = nn.Linear(self.LSTM_hidden_size * 4, self.LSTM_hidden_size * 2)
		self.decoder_i2d_attention_linear = nn.Linear(self.LSTM_hidden_size * 2, self.LSTM_hidden_size * 2)
		self.decoder_o2d_attention_linear = nn.Linear(self.LSTM_hidden_size * 2, self.LSTM_hidden_size * 2)
		self.decoder_d2op_attention_linear = nn.Linear(self.LSTM_hidden_size * 4, self.LSTM_hidden_size * 2)
		self.decoder_op2d_attention_linear = nn.Linear(self.LSTM_hidden_size * 2, self.LSTM_hidden_size * 2)
		self.encoder_self_attention_linear = nn.Linear(self.LSTM_hidden_size * 2, self.LSTM_hidden_size * 2)

		if self.operation_predictor:
			self.decoder_self_attention_linear = nn.Linear(self.LSTM_hidden_size * 6, self.LSTM_hidden_size * 2)
		else:
			self.decoder_self_attention_linear = nn.Linear(self.LSTM_hidden_size * 4, self.LSTM_hidden_size * 2)
		self.attention_tanh = nn.Tanh()

		self.keys = [int(x + self.value_offset) for x in range(-self.value_range, self.value_range + 1)]
		self.key_attention_linear = nn.Linear(self.embedding_size, self.embedding_size)

		self.addition_values = []

		for x in range(-self.value_range, self.value_range + 1):
			self.addition_values.append([])
			for y in range(-self.value_range, self.value_range + 1):
				self.addition_values[-1].append(int(x + y + self.value_offset))

		self.subtract_values = []

		for x in range(-self.value_range, self.value_range + 1):
			self.subtract_values.append([])
			for y in range(-self.value_range, self.value_range + 1):
				self.subtract_values[-1].append(int(y - x + self.value_offset))

		self.values = self.addition_values + self.subtract_values

		self.keys = np.array(self.keys)
		self.keys = data_utils.np_to_tensor(self.keys, 'int', self.cuda_flag, self.eval_flag)

		self.values = np.array(self.values)
		self.values = data_utils.np_to_tensor(self.values, 'int', self.cuda_flag, self.eval_flag)

		self.value_row_cnt = self.values.size()[0]
		self.value_col_cnt = self.values.size()[1]
		self.values = self.values.reshape(-1)
		self.operation_embedding = nn.Embedding(self.value_row_cnt, self.embedding_size)

		if self.use_properties:
			self.property_embedding = nn.Embedding(3, self.embedding_size)
			self.property_encoder = nn.LSTM(input_size=self.embedding_size, hidden_size=self.LSTM_hidden_size, num_layers=self.num_LSTM_layers, dropout=self.dropout_rate, bidirectional=True, batch_first=True)
			self.decoder_p2d_attention_linear = nn.Linear(self.LSTM_hidden_size * 2, self.LSTM_hidden_size * 2)
			self.decoder_d2p_attention_linear = nn.Linear(self.LSTM_hidden_size * 2, self.LSTM_hidden_size * 2)			

		if args.optimizer == 'adam':
			self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
		elif args.optimizer == 'sgd':
			self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
		elif args.optimizer == 'rmsprop':
			self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
		else:
			raise ValueError('optimizer undefined: ', args.optimizer)

	def init_weights(self, param_init):
		for param in self.parameters():
			nn.init.uniform_(param, -param_init, param_init)

	def lr_decay(self, lr_decay_rate):
		self.lr *= lr_decay_rate
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = self.lr

	def train_step(self):
		if self.gradient_clip > 0:
			clip_grad_norm(self.parameters(), self.gradient_clip)
		self.optimizer.step()

	def attention(self, encoder_outputs, decoder_output, encoder_attention_linear, decoder_attention_linear):
		"""
		[TODO] Compute the additive attention summary used between encoder and decoder streams.

		Input:
			encoder_outputs: tensor of shape (batch, source_len, hidden_dim).
			decoder_output: tensor of shape (batch, hidden_dim) or (batch, target_len, hidden_dim).
			encoder_attention_linear: projection module mapping encoder states to hidden_dim.
			decoder_attention_linear: projection module mapping decoder states to hidden_dim.

		Output:
			Tensor of shape (batch, hidden_dim) for a single decoder state, or a compatible
			batched attention representation when the decoder side has a time dimension.

"""
		pass


	def prog_exec(self, input_var_encoder_outputs, init_exec_hidden_state):
		"""
		[TODO] Run the latent program executor over encoded variable states.

		Input:
			input_var_encoder_outputs: tensor of shape (batch_io, var_len, 2 * hidden_size).
			init_exec_hidden_state: recurrent hidden state tuple compatible with prog_executor,
				each tensor shaped (2 * num_layers, batch_io, hidden_size).

		Output:
			Tensor of shape (batch_io, var_len, 2 * hidden_size) containing latent execution states.

"""
		pass

	def forward(self, batch_input_var_list, batch_output_var_list, batch_gt_code, batch_properties, eval_flag=False):
		"""
		[TODO] Run the full neural program synthesizer forward pass with optional latent execution.

		Input:
			batch_input_var_list: integer tensor of shape (batch, io_size, input_var_len)
				containing input variable values shifted into the embedding range.
			batch_output_var_list: integer tensor of shape (batch, io_size, output_var_len)
				containing target output variable values shifted into the embedding range.
			batch_gt_code: integer tensor of shape (batch, decode_len) containing target code tokens.
			batch_properties: integer tensor of shape (batch, property_len) for the property-signature baseline.
			eval_flag: boolean selecting autoregressive evaluation mode instead of teacher forcing.

		Output:
			Tuple (total_loss, logits, predictions):
			- total_loss: scalar tensor combining code prediction loss and enabled auxiliary losses.
			- logits: tensor of shape (batch, vocab_size, produced_decode_len).
			- predictions: tensor of shape (batch, produced_decode_len).

"""
		pass


if __name__ == "__main__":
	torch.manual_seed(42)
	np.random.seed(42)

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

	class TinyTokenizer:
		cls_token_id = 1
		eos_token_id = 2
		def __len__(self):
			return 9

	def make_tiny_model(latent_execution=False, operation_predictor=False, decoder_self_attention=False):
		model = CodeGenerator.__new__(CodeGenerator)
		nn.Module.__init__(model)
		model.cuda_flag = False
		model.eval_flag = False
		model.tokenizer = TinyTokenizer()
		model.vocab_size = len(model.tokenizer)
		model.batch_size = 2
		model.embedding_size = 6
		model.LSTM_hidden_size = 4
		model.MLP_hidden_size = 8
		model.num_LSTM_layers = 1
		model.num_MLP_layers = 1
		model.num_attention_layers = 1
		model.gradient_clip = 0.0
		model.lr = 0.001
		model.dropout_rate = 0.0
		model.max_input_len = 3
		model.max_decode_len = 4
		model.io_size = 2
		model.exec_period = 3
		model.dropout = nn.Dropout(p=0.0)
		model.ceLoss = nn.CrossEntropyLoss()
		model.mseLoss = nn.MSELoss()
		model.latent_execution = latent_execution
		model.no_partial_execution = False
		model.operation_predictor = operation_predictor
		model.value_range = 1
		model.value_offset = model.value_range * 4
		model.var_offset = -model.value_range + model.value_offset
		model.decoder_self_attention_flag = decoder_self_attention
		model.use_properties = False
		model.code_embedding = nn.Embedding(model.vocab_size, model.embedding_size)
		if model.decoder_self_attention_flag:
			model.code_predictor = MLPModel(model.num_MLP_layers, model.LSTM_hidden_size * 2, model.MLP_hidden_size, model.vocab_size, model.dropout_rate, model.cuda_flag)
		elif model.operation_predictor:
			model.code_predictor = MLPModel(model.num_MLP_layers, model.LSTM_hidden_size * 6, model.MLP_hidden_size, model.vocab_size, model.dropout_rate, model.cuda_flag)
		else:
			model.code_predictor = MLPModel(model.num_MLP_layers, model.LSTM_hidden_size * 4, model.MLP_hidden_size, model.vocab_size, model.dropout_rate, model.cuda_flag)
		model.var_embedding = nn.Embedding(model.value_offset * 2 + 1, model.embedding_size)
		model.input_var_encoder = nn.LSTM(input_size=model.embedding_size, hidden_size=model.LSTM_hidden_size, num_layers=model.num_LSTM_layers, dropout=model.dropout_rate, bidirectional=True, batch_first=True)
		model.output_var_encoder = nn.LSTM(input_size=model.embedding_size, hidden_size=model.LSTM_hidden_size, num_layers=model.num_LSTM_layers, dropout=model.dropout_rate, bidirectional=True, batch_first=True)
		model.operation_encoder = nn.LSTM(input_size=model.embedding_size, hidden_size=model.LSTM_hidden_size, num_layers=model.num_LSTM_layers, dropout=model.dropout_rate, bidirectional=True, batch_first=True)
		model.decoder = nn.LSTM(input_size=model.embedding_size, hidden_size=model.LSTM_hidden_size, num_layers=model.num_LSTM_layers, dropout=model.dropout_rate, bidirectional=True, batch_first=True)
		model.prog_executor = nn.LSTM(input_size=model.LSTM_hidden_size * 2, hidden_size=model.LSTM_hidden_size, num_layers=model.num_LSTM_layers, dropout=model.dropout_rate, bidirectional=True, batch_first=True)
		model.prog_executor_attention = nn.Linear(model.LSTM_hidden_size * 2, model.embedding_size)
		model.encoder_o2i_attention_linear = nn.ModuleList([nn.Linear(model.LSTM_hidden_size * 2, model.LSTM_hidden_size * 2) for _ in range(model.num_attention_layers)])
		model.encoder_i2o_attention_linear = nn.ModuleList([nn.Linear(model.LSTM_hidden_size * 2, model.LSTM_hidden_size * 2) for _ in range(model.num_attention_layers)])
		model.decoder_d2i_attention_linear = nn.Linear(model.LSTM_hidden_size * 2, model.LSTM_hidden_size * 2)
		model.decoder_d2o_attention_linear = nn.Linear(model.LSTM_hidden_size * 4, model.LSTM_hidden_size * 2)
		model.decoder_i2d_attention_linear = nn.Linear(model.LSTM_hidden_size * 2, model.LSTM_hidden_size * 2)
		model.decoder_o2d_attention_linear = nn.Linear(model.LSTM_hidden_size * 2, model.LSTM_hidden_size * 2)
		model.decoder_d2op_attention_linear = nn.Linear(model.LSTM_hidden_size * 4, model.LSTM_hidden_size * 2)
		model.decoder_op2d_attention_linear = nn.Linear(model.LSTM_hidden_size * 2, model.LSTM_hidden_size * 2)
		model.encoder_self_attention_linear = nn.Linear(model.LSTM_hidden_size * 2, model.LSTM_hidden_size * 2)
		if model.operation_predictor:
			model.decoder_self_attention_linear = nn.Linear(model.LSTM_hidden_size * 6, model.LSTM_hidden_size * 2)
		else:
			model.decoder_self_attention_linear = nn.Linear(model.LSTM_hidden_size * 4, model.LSTM_hidden_size * 2)
		model.attention_tanh = nn.Tanh()
		model.keys = data_utils.np_to_tensor([int(x + model.value_offset) for x in range(-model.value_range, model.value_range + 1)], 'int', model.cuda_flag, model.eval_flag)
		addition_values = []
		subtract_values = []
		for x in range(-model.value_range, model.value_range + 1):
			addition_values.append([])
			subtract_values.append([])
			for y in range(-model.value_range, model.value_range + 1):
				addition_values[-1].append(int(x + y + model.value_offset))
				subtract_values[-1].append(int(y - x + model.value_offset))
		model.values = data_utils.np_to_tensor(np.array(addition_values + subtract_values), 'int', model.cuda_flag, model.eval_flag)
		model.value_row_cnt = model.values.size()[0]
		model.value_col_cnt = model.values.size()[1]
		model.values = model.values.reshape(-1)
		model.key_attention_linear = nn.Linear(model.embedding_size, model.embedding_size)
		model.operation_embedding = nn.Embedding(model.value_row_cnt, model.embedding_size)
		return model

	print("=" * 70)
	print("latent-execution: CodeGenerator benchmark")
	print("Automated Test Suite - 3 ablated functions")
	print("=" * 70)
	print()

	print("-" * 60)
	print("[Test 1/3] CodeGenerator.attention")
	try:
		model = make_tiny_model()
		encoder_outputs = torch.randn(3, 5, 8)
		decoder_output = torch.randn(3, 8)
		y = model.attention(encoder_outputs, decoder_output, nn.Linear(8, 8), nn.Linear(8, 8))
		check("attention output not None", y is not None)
		if y is not None:
			check("attention output shape", tuple(y.shape) == (3, 8), f"got {tuple(y.shape)}")
			check("attention output finite", torch.isfinite(y).all().item())
			check("attention bounded by tanh", y.abs().max().item() <= 1.000001)
		else:
			skip_checks(3, "attention returned None")
	except Exception as exc:
		skip_checks(4, f"attention raised {type(exc).__name__}: {exc}")
	print()

	print("-" * 60)
	print("[Test 2/3] CodeGenerator.prog_exec")
	try:
		model = make_tiny_model()
		x = torch.randn(4, 3, 8)
		h0 = torch.randn(2, 4, 4)
		c0 = torch.randn(2, 4, 4)
		y = model.prog_exec(x, (h0, c0))
		check("prog_exec output not None", y is not None)
		if y is not None:
			check("prog_exec output shape", tuple(y.shape) == (4, 3, 8), f"got {tuple(y.shape)}")
			check("prog_exec output finite", torch.isfinite(y).all().item())
			check("prog_exec changes representation", not torch.allclose(y, x, atol=1e-5))
		else:
			skip_checks(3, "prog_exec returned None")
	except Exception as exc:
		skip_checks(4, f"prog_exec raised {type(exc).__name__}: {exc}")
	print()

	print("-" * 60)
	print("[Test 3/3] CodeGenerator.forward with operation predictor")
	try:
		model = make_tiny_model(latent_execution=False, operation_predictor=True, decoder_self_attention=False)
		batch_input_var_list = torch.tensor([[[3, 4, 5], [5, 4, 3]], [[4, 4, 4], [3, 5, 4]]], dtype=torch.long)
		batch_output_var_list = torch.tensor([[[4, 5, 6], [6, 5, 4]], [[5, 5, 5], [4, 6, 5]]], dtype=torch.long)
		batch_gt_code = torch.tensor([[1, 4, 3], [1, 5, 3]], dtype=torch.long)
		batch_properties = torch.zeros(2, 3, dtype=torch.long)
		total_loss, logits, predictions = model(batch_input_var_list, batch_output_var_list, batch_gt_code, batch_properties, eval_flag=False)
		check("forward outputs not None", total_loss is not None and logits is not None and predictions is not None)
		if total_loss is not None and logits is not None and predictions is not None:
			check("forward loss scalar finite", torch.is_tensor(total_loss) and total_loss.dim() == 0 and torch.isfinite(total_loss).item())
			check("forward logits shape", tuple(logits.shape) == (2, 9, 3), f"got {tuple(logits.shape)}")
			check("forward predictions shape", tuple(predictions.shape) == (2, 3), f"got {tuple(predictions.shape)}")
			check("forward predictions teacher forced", torch.equal(predictions, batch_gt_code))
			total_loss.backward()
			grad_exists = any(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters() if p.requires_grad)
			check("forward gradient path exists", grad_exists)
		else:
			skip_checks(5, "forward returned None")
	except Exception as exc:
		skip_checks(6, f"forward raised {type(exc).__name__}: {exc}")
	print()

	print("=" * 70)
	print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
	print("=" * 70)
	if failed == 0:
		print("All tests PASSED! The model code is complete and correct.")
	else:
		print(f"{failed} check(s) FAILED - some TODO functions may not be implemented correctly.")
	print("=" * 70)
	if failed:
		raise SystemExit(1)
