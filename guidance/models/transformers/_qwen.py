import torch
from ._transformers import Transformers
from loguru import logger


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
        **kwargs
    ):
        logger.info("Set tokenizer.eos_token_id = tokenizer.eod_id")
        tokenizer.eos_token_id = tokenizer.eod_id
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            echo=echo,
            caching=caching,
            temperature=temperature,
            compute_log_probs=compute_log_probs,
            device=device,
            **kwargs
        )


    def _get_logits(self, token_ids, forced_bytes):
        '''Computes the logits for the given token state.
        
        This overrides a method from the LocalEngine class that is used to get
        inference results from the model.
        forced_bytes is a list of bytes objects that will be forced to appear
        '''
        # make sure we don't run off the end of the model
        if len(token_ids) >= getattr(self.model_obj.config, "max_position_embeddings", 1e10):
            raise Exception(f"Attempted to run a transformers model past its maximum context window size of {self.model_obj.config.max_position_embeddings}!")

        # get the number of cache positions we are using
        cache_token_ids = self._cache_state["cache_token_ids"]
        num_cached = 0
        for id in cache_token_ids:
            if num_cached >= len(cache_token_ids) or num_cached >= len(token_ids) or token_ids[num_cached] != id:
                break
            num_cached += 1

        # reset the cache length according to that number of positions
        past_key_values = self._cache_state["past_key_values"]
        past_length = past_key_values[0][0].size(1) if past_key_values is not None else 0
        if past_length > num_cached:
            past_length = max(0, num_cached - 1) # note we recompute the last token because we don't bother to handle the special case of just computing logits
            self._cache_state["past_key_values"] = tuple(tuple(p[:, :past_length] for p in v) for v in past_key_values)
        cache_token_ids[past_length:] = []
        
        # call the model
        new_token_ids = token_ids[past_length:]
        if len(new_token_ids) > 0:
            with torch.no_grad():
                model_out = self.model_obj(
                    input_ids=torch.tensor(new_token_ids).unsqueeze(0).to(self.device),
                    past_key_values=self._cache_state["past_key_values"],
                    use_cache=True,
                    position_ids=torch.arange(past_length, past_length+len(new_token_ids)).unsqueeze(0).to(self.device),
                    attention_mask=torch.ones(1, past_length + len(new_token_ids)).to(self.device),
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False
                )

            # save the results
            self._cache_state["past_key_values"] = model_out.past_key_values
            cache_token_ids.extend(new_token_ids)
            self._cache_state["logits"] = model_out.logits[0, -1, :].cpu().numpy()
        
        ret = self._cache_state["logits"]
        assert ret is not None, "Something went wrong with the cache!"
        return ret
    