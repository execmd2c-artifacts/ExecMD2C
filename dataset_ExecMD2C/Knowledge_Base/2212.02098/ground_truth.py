# ============================================================
# ground_truth.py - Explicit Memory DQN LSTM Core Model Components
# Source: Knowledge_Base/explicit-memory-main/agent/dqn/nn/lstm.py
#        Knowledge_Base/explicit-memory-main/agent/utils.py
#
# Contains ONLY the model architecture definitions and direct dependencies.
# No training, inference, dataset, environment, replay-buffer, or pipeline code.
# ============================================================

from copy import deepcopy
from typing import Literal, Union

import logging
import numpy as np
import torch
from torch import nn


# --- [Original file: agent/utils.py] ---
def split_by_possessive(name_entity: str) -> tuple[str, str]:
    """Separate name and entity from the given string.

    Args:
        name_entity: e.g., "tae's laptop"

    Returns:
        name: e.g., tae
        entity: e.g., laptop

    """
    logging.debug(f"spliting name and entity from {name_entity}")
    if "'s " in name_entity:
        name, entity = name_entity.split("'s ")
    else:
        name, entity = None, None

    return name, entity


def positional_encoding(
    positions: int,
    dimensions: int,
    scaling_factor: float = 10000.0,
    return_tensor: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Generate sinusoidal positional encoding.

    Parameters:
    positions (int): The number of positions in the sequence.
    dimensions (int): The dimension of the embedding vectors.
    scaling_factor (float): The scaling factor used in the sinusoidal functions.
    return_tensor (bool): If True, return a PyTorch tensor; otherwise, return a NumPy array.

    Returns:
    Union[np.ndarray, torch.Tensor]: A positional encoding in the form of either a NumPy array or a PyTorch tensor.
    """
    # Ensure the number of dimensions is even
    assert dimensions % 2 == 0, "The dimension must be even."

    # Initialize a matrix of position encodings with zeros
    pos_enc = np.zeros((positions, dimensions))

    # Compute the positional encodings
    for pos in range(positions):
        for i in range(0, dimensions, 2):
            pos_enc[pos, i] = np.sin(pos / (scaling_factor ** ((2 * i) / dimensions)))
            pos_enc[pos, i + 1] = np.cos(
                pos / (scaling_factor ** ((2 * (i + 1)) / dimensions))
            )

    # Return as PyTorch tensor if requested
    if return_tensor:
        return torch.from_numpy(pos_enc)

    return pos_enc


# --- [Original file: agent/dqn/nn/lstm.py] ---
class LSTM(nn.Module):
    """A simple LSTM network."""

    def __init__(
        self,
        capacity: dict,
        entities: list,
        relations: list,
        n_actions: int,
        memory_of_interest: list,
        hidden_size: int = 64,
        num_layers: int = 2,
        embedding_dim: int = 64,
        make_categorical_embeddings: bool = False,
        batch_first: bool = True,
        device: str = "cpu",
        include_human: Literal["sum", "concat", None] = "sum",
        human_embedding_on_object_location: bool = False,
        dueling_dqn: bool = False,
        fuse_information: Literal["concat", "sum"] = "sum",
        include_positional_encoding: bool = True,
        max_timesteps: int | None = None,
        max_strength: int | None = None,
        is_actor: bool = False,
        is_critic: bool = False,
    ) -> None:
        """Initialize the LSTM.

        Args:
            capacity: the capacities of memory systems. e.g., {"episodic": 16,
                "semantic": 16, "short": 1}
            entities: list of entities, e.g., ["Foo", "Bar", "laptop", "phone",
                "desk", "lap"]
            relations : list of relations, e.g., ["atlocation", "north", "south"]
            n_actions: number of actions. This should be 3, at the moment.
            memory_of_interest: e.g., ["episodic", "semantic", "short"]
            hidden_size: hidden size of the LSTM
            num_layers: number of the LSTM layers
            embedding_dim: entity embedding dimension (e.g., 32)
            make_categorical_embeddings: whether to use categorical embeddings or not.
            batch_first: Should the batch dimension be the first or not.
            device: "cpu" or "cuda"
            include_human:
                None: Don't include humans
                "sum": sum up the human embeddings with object / object_location
                    embeddings.
                "cocnat": concatenate the human embeddings to object /
                    object_location embeddings.
            human_embedding_on_object_location: whether to superposition the human
                embedding on the tail (object location entity).
            dueling_dqn: whether to use dueling DQN or not.
            fuse_information: "concat" or "sum"
            include_positional_encoding: whether to include the number 4, i.e.,
                strength or timestamp in the entity list.
            max_timesteps: maximum number of timesteps. This is only used when
                `include_positional_encoding` is True.
            max_strength: maximum strength. This is only used when
                `include_positional_encoding` is True.

        """
        super().__init__()
        self.capacity = capacity
        self.memory_of_interest = memory_of_interest
        self.entities = entities
        self.relations = relations
        self.n_actions = n_actions
        self.embedding_dim = embedding_dim
        self.make_categorical_embeddings = make_categorical_embeddings
        self.device = device
        self.include_human = include_human
        self.human_embedding_on_object_location = human_embedding_on_object_location
        self.dueling_dqn = dueling_dqn
        self.fuse_information = fuse_information
        self.include_positional_encoding = include_positional_encoding
        self.max_timesteps = max_timesteps
        self.max_strength = max_strength
        self.is_actor = is_actor
        self.is_critic = is_critic

        if self.fuse_information == "concat":
            self.linear_layer_hidden_size = hidden_size * len(self.memory_of_interest)
        elif self.fuse_information == "sum":
            self.linear_layer_hidden_size = hidden_size
        else:
            raise ValueError(
                f"fuse_information should be one of 'concat' or 'sum', but "
                f"{self.fuse_information} was given!"
            )

        if self.include_positional_encoding:
            assert self.max_timesteps is not None
            assert self.max_strength is not None
            self.positional_encoding = positional_encoding(
                positions=max(self.max_timesteps, self.max_strength) + 1,
                dimensions=self.linear_layer_hidden_size,
                scaling_factor=10000,
                return_tensor=True,
            ).to(self.device)

        self.create_embeddings()
        if "episodic" in self.memory_of_interest:
            self.lstm_e = nn.LSTM(
                self.input_size_e,
                hidden_size,
                num_layers,
                batch_first=batch_first,
                device=self.device,
            )
            if self.fuse_information == "concat":
                self.fc_e0 = nn.Linear(hidden_size, hidden_size, device=self.device)
                self.fc_e1 = nn.Linear(hidden_size, hidden_size, device=self.device)

        if "episodic_agent" in self.memory_of_interest:
            self.lstm_e_agent = nn.LSTM(
                self.input_size_e,
                hidden_size,
                num_layers,
                batch_first=batch_first,
                device=self.device,
            )
            if self.fuse_information == "concat":
                self.fc_e0_agent = nn.Linear(
                    hidden_size, hidden_size, device=self.device
                )
                self.fc_e1_agent = nn.Linear(
                    hidden_size, hidden_size, device=self.device
                )

        if "semantic" in self.memory_of_interest:
            self.lstm_s = nn.LSTM(
                self.input_size_s,
                hidden_size,
                num_layers,
                batch_first=batch_first,
                device=self.device,
            )
            if self.fuse_information == "concat":
                self.fc_s0 = nn.Linear(hidden_size, hidden_size, device=self.device)
                self.fc_s1 = nn.Linear(hidden_size, hidden_size, device=self.device)

        if "semantic_map" in self.memory_of_interest:
            self.lstm_s_map = nn.LSTM(
                self.input_size_s,
                hidden_size,
                num_layers,
                batch_first=batch_first,
                device=self.device,
            )
            if self.fuse_information == "concat":
                self.fc_s0_map = nn.Linear(hidden_size, hidden_size, device=self.device)
                self.fc_s1_map = nn.Linear(hidden_size, hidden_size, device=self.device)

        if "short" in self.memory_of_interest:
            self.lstm_o = nn.LSTM(
                self.input_size_o,
                hidden_size,
                num_layers,
                batch_first=batch_first,
                device=self.device,
            )
            if self.fuse_information == "concat":
                self.fc_o0 = nn.Linear(hidden_size, hidden_size, device=self.device)
                self.fc_o1 = nn.Linear(hidden_size, hidden_size, device=self.device)

        self.advantage_layer = nn.Sequential(
            nn.Linear(
                self.linear_layer_hidden_size,
                self.linear_layer_hidden_size,
                device=self.device,
            ),
            nn.ReLU(),
            nn.Linear(
                self.linear_layer_hidden_size,
                self.n_actions,
                device=self.device,
            ),
        )

        if self.dueling_dqn:
            self.value_layer = nn.Sequential(
                nn.Linear(
                    self.linear_layer_hidden_size,
                    self.linear_layer_hidden_size,
                    device=self.device,
                ),
                nn.ReLU(),
                nn.Linear(
                    self.linear_layer_hidden_size,
                    1,
                    device=self.device,
                ),
            )

        self.relu = nn.ReLU()

    def create_embeddings(self) -> None:
        """Create learnable embeddings."""
        if isinstance(self.entities, dict):
            self.word2idx = (
                ["<PAD>"]
                + [name for names in self.entities.values() for name in names]
                + self.relations
            )
        elif isinstance(self.entities, list):
            self.word2idx = ["<PAD>"] + self.entities + self.relations
        else:
            raise ValueError(
                "entities should be either a list or a dictionary, but "
                f"{type(self.entities)} was given!"
            )
        self.word2idx = {word: idx for idx, word in enumerate(self.word2idx)}

        self.embeddings = nn.Embedding(
            len(self.word2idx),
            self.embedding_dim,
            device=self.device,
            padding_idx=0,
        )

        if self.make_categorical_embeddings:
            # Assuming self.entities is a dictionary where keys are categories and
            # values are lists of entity names

            # Create a dictionary to keep track of starting index for each category
            category_start_indices = {}
            current_index = 1  # Start from 1 to skip the <PAD> token
            for category, names in self.entities.items():
                category_start_indices[category] = current_index
                current_index += len(names)

            # Re-initialize embeddings by category
            for category, start_idx in category_start_indices.items():
                end_idx = start_idx + len(self.entities[category])
                init_vector = torch.randn(self.embedding_dim, device=self.device)
                self.embeddings.weight.data[start_idx:end_idx] = init_vector.repeat(
                    end_idx - start_idx, 1
                )
            # Note: Relations are not re-initialized by category, assuming they are
            # separate from entities

        if self.fuse_information == "concat":

            self.input_size_s = self.embedding_dim * 2
            if (self.include_human is None) or (self.include_human == "sum"):
                self.input_size_e = self.embedding_dim * 2
                self.input_size_o = self.embedding_dim * 2

            elif self.include_human == "concat":
                raise ValueError("This is deprecated!")
                # self.input_size_e = self.embedding_dim * 3
                # self.input_size_o = self.embedding_dim * 3
            else:
                raise ValueError(
                    "include_human should be one of None or 'sum'"
                    f"but {self.include_human} was given!"
                )

        elif self.fuse_information == "sum":
            self.input_size_s = self.embedding_dim
            self.input_size_e = self.embedding_dim
            self.input_size_o = self.embedding_dim
        else:
            raise ValueError(
                f"fuse_information should be one of 'concat' or 'sum', but "
                f"{self.fuse_information} was given!"
            )

    def make_embedding(self, mem: list[str], memory_type: str) -> torch.Tensor:
        """Create one embedding vector with summation and concatenation.

        Args:
            mem: memory as a quadruple: [head, relation, tail, num]
            memory_type: "episodic", "semantic", or "short"

        Returns:
            one embedding vector made from one memory element.

        """
        if mem == ["<PAD>", "<PAD>", "<PAD>", "<PAD>"]:
            if self.fuse_information == "sum":
                padding_embedding = self.embeddings(
                    torch.tensor(self.word2idx["<PAD>"], device=self.device)
                )
                return padding_embedding
            else:
                final_embedding = torch.concat(
                    [
                        self.embeddings(
                            torch.tensor(self.word2idx["<PAD>"], device=self.device)
                        ),
                        self.embeddings(
                            torch.tensor(self.word2idx["<PAD>"], device=self.device)
                        ),
                    ]
                )
                return final_embedding

        else:
            if memory_type == "semantic":
                obj = mem[0]
            else:
                human, obj = split_by_possessive(mem[0])

            obj_loc = mem[2]

        object_embedding = self.embeddings(
            torch.tensor(self.word2idx[obj], device=self.device)
        )
        object_location_embedding = self.embeddings(
            torch.tensor(self.word2idx[obj_loc], device=self.device)
        )

        if memory_type == "semantic":
            if self.fuse_information == "concat":
                final_embedding = torch.concat(
                    [object_embedding, object_location_embedding]
                )
            elif self.fuse_information == "sum":
                final_embedding = object_embedding + object_location_embedding
            else:
                raise ValueError(
                    f"fuse_information should be one of 'concat' or 'sum', but "
                    f"{self.fuse_information} was given!"
                )

        elif memory_type in ["episodic", "short"]:
            human_embedding = self.embeddings(
                torch.tensor(self.word2idx[human], device=self.device)
            )

            if self.include_human is None:
                if self.fuse_information == "concat":
                    final_embedding = torch.concat(
                        [object_embedding, object_location_embedding]
                    )
                elif self.fuse_information == "sum":
                    final_embedding = object_embedding + object_location_embedding
                else:
                    raise ValueError(
                        f"fuse_information should be one of 'concat' or 'sum', but "
                        f"{self.fuse_information} was given!"
                    )
            elif self.include_human == "sum":
                if self.fuse_information == "concat":
                    final_embedding = [object_embedding + human_embedding]

                    if self.human_embedding_on_object_location:
                        raise ValueError("This is deprecated!")
                    else:
                        final_embedding.append(object_location_embedding)

                    final_embedding = torch.concat(final_embedding)
                elif self.fuse_information == "sum":
                    final_embedding = (
                        object_embedding + object_location_embedding + human_embedding
                    )

            elif self.include_human == "concat":
                raise ValueError("This is deprecated!")
        else:
            raise ValueError

        if self.include_positional_encoding:
            final_embedding += self.positional_encoding[mem[3]]

        return final_embedding

    def create_batch(self, x: list[list[list]], memory_type: str) -> torch.Tensor:
        """Create one batch from data.

        Args:
            x: a batch of episodic, semantic, or short memories.
            memory_type: "episodic", "semantic", or "short"

        Returns:
            batch of embeddings.

        """
        mem_pad = ["<PAD>", "<PAD>", "<PAD>", "<PAD>"]

        for mems in x:
            for _ in range(self.capacity[memory_type] - len(mems)):
                # this is a dummy entry for padding.
                mems.append(mem_pad)
        batch_embeddings = []
        for mems in x:
            embeddings = []
            for mem in mems:
                mem_emb = self.make_embedding(mem, memory_type)
                embeddings.append(mem_emb)
            embeddings = torch.stack(embeddings)
            batch_embeddings.append(embeddings)

        batch_embeddings = torch.stack(batch_embeddings)

        return batch_embeddings

    def forward(self, x_: np.ndarray) -> torch.Tensor:
        """Forward-pass.

        Note that before we make a forward pass, argument x_ will be deepcopied. This
        is because we will modify x_ in the forward pass, and we don't want to modify
        the original x_. This slows down the process, but it's necessary.

        Args:
            x: a batch of memories. Each element of the batch is a np.ndarray of dict
            memories. x being a np.ndarray speeds up the process.

        Returns:
            Q-values, (action, distribution), or value.

        """
        x = deepcopy(x_)
        assert isinstance(x, np.ndarray)
        to_concat = []
        if "episodic" in self.memory_of_interest:
            batch_e = [sample["episodic"] for sample in x]  # sample is a dict
            batch_e = self.create_batch(batch_e, memory_type="episodic")
            lstm_out_e, _ = self.lstm_e(batch_e)

            if self.fuse_information == "concat":
                fc_out_e = self.relu(
                    self.fc_e1(self.relu(self.fc_e0(lstm_out_e[:, -1, :])))
                )
                to_concat.append(fc_out_e)
            else:
                to_concat.append(lstm_out_e[:, -1, :])

        if "episodic_agent" in self.memory_of_interest:
            batch_e_agent = [sample["episodic_agent"] for sample in x]
            batch_e_agent = self.create_batch(
                batch_e_agent, memory_type="episodic_agent"
            )
            lstm_out_e_agent, _ = self.lstm_e_agent(batch_e_agent)

            if self.fuse_information == "concat":
                fc_out_e_agent = self.relu(
                    self.fc_e1_agent(
                        self.relu(self.fc_e0_agent(lstm_out_e_agent[:, -1, :]))
                    )
                )
                to_concat.append(fc_out_e_agent)
            else:
                to_concat.append(lstm_out_e_agent[:, -1, :])

        if "semantic" in self.memory_of_interest:
            batch_s = [sample["semantic"] for sample in x]
            batch_s = self.create_batch(batch_s, memory_type="semantic")
            lstm_out_s, _ = self.lstm_s(batch_s)

            if self.fuse_information == "concat":
                fc_out_s = self.relu(
                    self.fc_s1(self.relu(self.fc_s0(lstm_out_s[:, -1, :])))
                )
                to_concat.append(fc_out_s)
            else:
                to_concat.append(lstm_out_s[:, -1, :])

        if "semantic_map" in self.memory_of_interest:
            batch_s_map = [sample["semantic_map"] for sample in x]
            batch_s_map = self.create_batch(batch_s_map, memory_type="semantic_map")
            lstm_out_s_map, _ = self.lstm_s_map(batch_s_map)

            if self.fuse_information == "concat":
                fc_out_s_map = self.relu(
                    self.fc_s1_map(self.relu(self.fc_s0_map(lstm_out_s_map[:, -1, :])))
                )
                to_concat.append(fc_out_s_map)
            else:
                to_concat.append(lstm_out_s_map[:, -1, :])

        if "short" in self.memory_of_interest:
            batch_o = [sample["short"] for sample in x]
            batch_o = self.create_batch(batch_o, memory_type="short")
            lstm_out_o, _ = self.lstm_o(batch_o)

            if self.fuse_information == "concat":
                fc_out_o = self.relu(
                    self.fc_o1(self.relu(self.fc_o0(lstm_out_o[:, -1, :])))
                )
                to_concat.append(fc_out_o)
            else:
                to_concat.append(lstm_out_o[:, -1, :])

        if self.fuse_information == "concat":
            fc_out_all = torch.concat(to_concat, dim=-1)
        elif self.fuse_information == "sum":
            fc_out_all = torch.sum(torch.stack(to_concat), dim=0)
        else:
            raise ValueError(
                f"fuse_information should be one of 'concat' or 'sum', but "
                f"{self.fuse_information} was given!"
            )

        if self.dueling_dqn:
            value = self.value_layer(fc_out_all)
            advantage = self.advantage_layer(fc_out_all)
            q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        else:
            q = self.advantage_layer(fc_out_all)

        return q


def _sample_states() -> np.ndarray:
    return np.array(
        [
            {
                "episodic": [["Alice's key", "at", "kitchen", 1]],
                "semantic": [["key", "at", "kitchen", 2]],
                "short": [["Bob's apple", "at", "garden", 3]],
            },
            {
                "episodic": [
                    ["Bob's key", "at", "garden", 2],
                    ["Alice's apple", "at", "kitchen", 3],
                ],
                "semantic": [],
                "short": [],
            },
        ],
        dtype=object,
    )


def _make_model(
    fuse_information: Literal["concat", "sum"] = "sum",
    dueling_dqn: bool = True,
    include_positional_encoding: bool = True,
) -> LSTM:
    return LSTM(
        capacity={"episodic": 2, "semantic": 2, "short": 1},
        entities=["Alice", "Bob", "key", "apple", "kitchen", "garden"],
        relations=[],
        n_actions=3,
        memory_of_interest=["episodic", "semantic", "short"],
        hidden_size=8,
        num_layers=1,
        embedding_dim=8,
        include_human="sum",
        dueling_dqn=dueling_dqn,
        fuse_information=fuse_information,
        include_positional_encoding=include_positional_encoding,
        max_timesteps=8,
        max_strength=8,
        device="cpu",
    )


# ============================================================
# __main__: Automated test suite for 3 ablated functions
# ============================================================

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

    print("=" * 70)
    print("Explicit Memory: DQN LSTM Core Model Tests")
    print("Automated Test Suite - 3 ablated functions")
    print("=" * 70)
    print()

    device = torch.device("cpu")
    print(f"Device: {device}")
    print()

    # ==========================================================
    # Test 1/3: LSTM.make_embedding
    # ==========================================================
    print("-" * 60)
    print("[Test 1/3] LSTM.make_embedding - explicit memory element encoding")
    make_embedding_checks = 9
    try:
        model = _make_model(fuse_information="sum", include_positional_encoding=True)
        episodic_embedding = model.make_embedding(
            ["Alice's key", "at", "kitchen", 1], "episodic"
        )
        semantic_embedding = model.make_embedding(["key", "at", "kitchen", 1], "semantic")
        semantic_embedding_later = model.make_embedding(
            ["key", "at", "kitchen", 2], "semantic"
        )
        padding_embedding = model.make_embedding(
            ["<PAD>", "<PAD>", "<PAD>", "<PAD>"], "semantic"
        )

        check("make_embedding episodic output not None", episodic_embedding is not None)
        if episodic_embedding is not None:
            expected_shape = (8,)
            check("make_embedding episodic shape", tuple(episodic_embedding.shape) == expected_shape,
                  f"expected {expected_shape}, got {tuple(episodic_embedding.shape)}")
            check("make_embedding episodic finite", torch.isfinite(episodic_embedding).all().item())
            check("make_embedding semantic shape", tuple(semantic_embedding.shape) == expected_shape,
                  f"expected {expected_shape}, got {tuple(semantic_embedding.shape)}")
            check("make_embedding padding shape", tuple(padding_embedding.shape) == expected_shape,
                  f"expected {expected_shape}, got {tuple(padding_embedding.shape)}")
            check("make_embedding human branch changes episodic memory",
                  not torch.allclose(episodic_embedding, semantic_embedding, atol=1e-6),
                  "episodic memory should include the human entity when include_human='sum'")
            check("make_embedding positional encoding changes timestamp",
                  not torch.allclose(semantic_embedding, semantic_embedding_later, atol=1e-6),
                  "different memory strengths/timestamps should affect the embedding")
            check("make_embedding padding equals pad token",
                  torch.allclose(
                      padding_embedding,
                      model.embeddings(torch.tensor(model.word2idx["<PAD>"], device=device)),
                      atol=1e-6,
                  ))
            loss = episodic_embedding.sum() + semantic_embedding.sum()
            loss.backward()
            alice_idx = model.word2idx["Alice"]
            key_idx = model.word2idx["key"]
            check("make_embedding selected embeddings receive gradient",
                  model.embeddings.weight.grad is not None
                  and model.embeddings.weight.grad[alice_idx].abs().sum().item() > 0
                  and model.embeddings.weight.grad[key_idx].abs().sum().item() > 0)
        else:
            skip_checks(make_embedding_checks - 1, "make_embedding returned None")
    except Exception as exc:
        skip_checks(make_embedding_checks, f"LSTM.make_embedding raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 2/3: LSTM.create_batch
    # ==========================================================
    print("-" * 60)
    print("[Test 2/3] LSTM.create_batch - capacity padding and batch construction")
    create_batch_checks = 7
    try:
        model = _make_model(fuse_information="sum", include_positional_encoding=True)
        episodic_batch_source = [
            [["Alice's key", "at", "kitchen", 1]],
            [["Bob's apple", "at", "garden", 2], ["Alice's apple", "at", "kitchen", 3]],
        ]
        episodic_batch = model.create_batch(episodic_batch_source, "episodic")
        semantic_batch_source = [[["key", "at", "kitchen", 1]], []]
        semantic_batch = model.create_batch(semantic_batch_source, "semantic")

        check("create_batch episodic output not None", episodic_batch is not None)
        if episodic_batch is not None:
            expected_episode_shape = (2, 2, 8)
            expected_semantic_shape = (2, 2, 8)
            check("create_batch episodic shape", tuple(episodic_batch.shape) == expected_episode_shape,
                  f"expected {expected_episode_shape}, got {tuple(episodic_batch.shape)}")
            check("create_batch semantic shape", tuple(semantic_batch.shape) == expected_semantic_shape,
                  f"expected {expected_semantic_shape}, got {tuple(semantic_batch.shape)}")
            check("create_batch episodic finite", torch.isfinite(episodic_batch).all().item())
            check("create_batch semantic finite", torch.isfinite(semantic_batch).all().item())
            check("create_batch pads each sample to capacity",
                  all(len(sample) == model.capacity["episodic"] for sample in episodic_batch_source)
                  and all(len(sample) == model.capacity["semantic"] for sample in semantic_batch_source))
            pad_row = model.make_embedding(["<PAD>", "<PAD>", "<PAD>", "<PAD>"], "semantic")
            check("create_batch padded row uses pad embedding",
                  torch.allclose(semantic_batch[1, 0], pad_row, atol=1e-6)
                  and torch.allclose(semantic_batch[1, 1], pad_row, atol=1e-6))
        else:
            skip_checks(create_batch_checks - 1, "create_batch returned None")
    except Exception as exc:
        skip_checks(create_batch_checks, f"LSTM.create_batch raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 3/3: LSTM.forward
    # ==========================================================
    print("-" * 60)
    print("[Test 3/3] LSTM.forward - multi-memory Q-value assembly")
    forward_checks = 10
    try:
        model = _make_model(fuse_information="sum", dueling_dqn=True, include_positional_encoding=True)
        model.train()
        states = _sample_states()
        original_lengths = [
            {key: len(value) for key, value in sample.items()}
            for sample in states.tolist()
        ]
        q_values = model(states)

        check("forward output not None", q_values is not None)
        if q_values is not None:
            expected_q_shape = (2, 3)
            check("forward q-value shape", tuple(q_values.shape) == expected_q_shape,
                  f"expected {expected_q_shape}, got {tuple(q_values.shape)}")
            check("forward q-values finite", torch.isfinite(q_values).all().item())
            check("forward preserves input memory lengths",
                  original_lengths == [
                      {key: len(value) for key, value in sample.items()}
                      for sample in states.tolist()
                  ],
                  "forward should not mutate the caller's memory state")
            check("forward dueling value layer exists", hasattr(model, "value_layer"))
            loss = q_values.sum()
            loss.backward()
            check("forward embedding gradients exist",
                  model.embeddings.weight.grad is not None
                  and torch.isfinite(model.embeddings.weight.grad).all().item()
                  and model.embeddings.weight.grad.abs().sum().item() > 0)
            check("forward episodic LSTM gradients exist",
                  model.lstm_e.weight_ih_l0.grad is not None
                  and model.lstm_e.weight_ih_l0.grad.abs().sum().item() > 0)
            check("forward semantic LSTM gradients exist",
                  model.lstm_s.weight_ih_l0.grad is not None
                  and model.lstm_s.weight_ih_l0.grad.abs().sum().item() > 0)
            check("forward short-memory LSTM gradients exist",
                  model.lstm_o.weight_ih_l0.grad is not None
                  and model.lstm_o.weight_ih_l0.grad.abs().sum().item() > 0)

            concat_model = _make_model(
                fuse_information="concat",
                dueling_dqn=False,
                include_positional_encoding=False,
            )
            concat_q_values = concat_model(states)
            check("forward concat fusion q-value shape", tuple(concat_q_values.shape) == expected_q_shape,
                  f"expected {expected_q_shape}, got {tuple(concat_q_values.shape)}")
        else:
            skip_checks(forward_checks - 1, "forward returned None")
    except Exception as exc:
        skip_checks(forward_checks, f"LSTM.forward raised {type(exc).__name__}: {exc}")
    print()

    print("=" * 70)
    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    print("=" * 70)
    if failed == 0:
        print("All tests PASSED! The model code is complete and correct.")
    else:
        print(f"{failed} check(s) FAILED - some [TODO] functions may not be implemented correctly.")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
