import torch
from tqdm import tqdm
from ._transformers import Transformers
from ._transformers import TransformersChat
from loguru import logger

import time


class Qwen(Transformers):
    def __init__(
        self,
        model=None,
        tokenizer=None,
        echo=True,
        caching=True,
        temperature=0.0,
        compute_log_probs=False,
        device=None,
        **kwargs,
    ):
        logger.info("Set tokenizer.eos_token_id = tokenizer.eod_id")
        tokenizer.eos_token_id = tokenizer.eod_id

        self.generated_logits = []
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            echo=echo,
            caching=caching,
            temperature=temperature,
            compute_log_probs=compute_log_probs,
            device=device,
            **kwargs,
        )
        self.time_table = {}

    def decode(self, ids):
        return self._orig_tokenizer.decode(ids)

    def encode(self, text):
        return self._orig_tokenizer.encode(text)

    def _get_logits(self, token_ids, forced_bytes, current_temp, **kwargs):
        """Computes the logits for the given token state.

        This overrides a method from the LocalEngine class that is used to get
        inference results from the model.
        """
        # make sure we don't run off the end of the model

        _text = self.decode(token_ids)
        # new_token_ids = self.encode(_text)
        # assert len(new_token_ids) == len(token_ids)

        if len(token_ids) >= getattr(
            self.model_obj.config, "max_position_embeddings", 1e10
        ):
            raise Exception(
                f"Attempted to run a transformers model past its maximum context window size of {self.model_obj.config.max_position_embeddings}!"
            )

        # get the number of cache positions we are using
        # import ipdb; ipdb.set_trace()
        cache_token_ids = self._cache_state["cache_token_ids"]
        num_cached = 0
        for id in cache_token_ids:
            if (
                num_cached >= len(cache_token_ids)
                or num_cached >= len(token_ids)
                or token_ids[num_cached] != id
            ):
                break
            num_cached += 1

        # reset the cache length according to that number of positions
        past_key_values = self._cache_state["past_key_values"]
        past_length = (
            past_key_values[0][0].size(1) if past_key_values is not None else 0
        )
        if past_length > num_cached:
            past_length = max(
                0, num_cached - 1
            )  # note we recompute the last token because we don't bother to handle the special case of just computing logits
            self._cache_state["past_key_values"] = tuple(
                tuple(p[:, :past_length] for p in v) for v in past_key_values
            )
        cache_token_ids[past_length:] = []

        # call the model
        new_token_ids = token_ids[past_length:]
        # print('Past length', past_length, 'new length', len(new_token_ids))
        if len(new_token_ids)==0:
            import ipdb; ipdb.set_trace()
        if len(new_token_ids) > 0:
            with torch.no_grad():
                model_out = self.model_obj(
                    input_ids=torch.tensor(new_token_ids).unsqueeze(0).to(self.device),
                    past_key_values=self._cache_state["past_key_values"],
                    use_cache=False,
                    position_ids=torch.arange(
                        past_length, past_length + len(new_token_ids)
                    )
                    .unsqueeze(0)
                    .to(self.device),
                    attention_mask=torch.ones(1, past_length + len(new_token_ids)).to(
                        self.device
                    ),
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )

            # save the results
            self._cache_state["past_key_values"] = model_out.past_key_values
            cache_token_ids.extend(new_token_ids)
            logits = model_out.logits[0, -1, :].float()
            # if self.compute_log_probs:
                # logits = logits.softmax(-1)
            self._cache_state["logits"] = logits.cpu().numpy()
            scores, ids = model_out.logits[0, -1, :].softmax(-1).topk(2)
            topk = {}
            for i in range(len(ids)):
                topk[self.decode([ids[i].item()])] = scores[i].item()
            # print("topk", topk)
            # import ipdb;ipdb.set_trace()  # pred_token = self.decode(x[1])

        return self._cache_state["logits"]

    def _cleanup_tokens(self, token_ids, token_byte_positions):
        return token_ids, token_byte_positions

    def _tokenize_prefix(self, byte_string):
        string = str(byte_string, encoding="utf8")
        # if '>assistant' in string:
        #     last_idx_of_assistant = string.rfind('>assistant')
        #     prefix = string[:last_idx_of_assistant]
            # prefix, assistant_prefix = string.split('>assistant')
            
        token_ids = self.encode(string)
        bytes_position = []

        for i in range(len(token_ids)):
            _ids = token_ids[: i + 1]
            _s = self.decode(_ids)
            _bytes = bytes(_s, encoding="utf8")
            bytes_position.append(len(_bytes))
            
        posible_end_tokens = []
        if len(bytes_position):
            last_byte = _bytes[bytes_position[-2]:]
            for token in self.tokens:
                if token.startswith(last_byte):
                    posible_end_tokens.append(token)
            if len(posible_end_tokens):
                return token_ids[: -1], bytes_position[: -1]
        return token_ids, bytes_position
    
class QwenChat(Qwen):
    pass