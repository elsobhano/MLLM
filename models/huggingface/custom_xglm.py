from transformers import XGLMForCausalLM
import torch
from typing import Optional, Dict, Any

class CustomXGLMForCausalLM(XGLMForCausalLM):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        custom_model = cls(model.config)
        custom_model.load_state_dict(model.state_dict())
        for attr_name, attr_value in model.__dict__.items():
            if not hasattr(custom_model, attr_name):
                setattr(custom_model, attr_name, attr_value)
        return custom_model

    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[tuple] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Override prepare_inputs_for_generation to handle input embeddings
        """
        # First generation step
        if past_key_values is None:
            if inputs_embeds is not None:
                model_inputs = {
                    "inputs_embeds": inputs_embeds,
                }
            else:
                model_inputs = {
                    "input_ids": input_ids,
                }
        # Subsequent generation steps
        else:
            if inputs_embeds is not None:
                last_token_embed = self.get_input_embeddings()(input_ids[:, -1:])
                model_inputs = {
                    "inputs_embeds": last_token_embed,
                }
            else:
                model_inputs = {
                    "input_ids": input_ids[:, -1:],
                }

        # Common inputs for both first and subsequent steps
        model_inputs.update({
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "position_ids": kwargs.get("position_ids", None),
        })

        return model_inputs

    def generate(self, inputs_embeds=None, **generate_kwargs):
        if inputs_embeds is not None:
            # Generate a dummy input_ids tensor for compatibility
            batch_size = inputs_embeds.shape[0]
            seq_len = inputs_embeds.shape[1]
            input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=inputs_embeds.device)

            # Pass inputs_embeds to the parent method
            return super().generate(
                input_ids=input_ids,  # Dummy input_ids
                inputs_embeds=inputs_embeds,  # Use embeddings
                **generate_kwargs,  # Include all other generation parameters
            )
        else:
            # Fallback to default behavior
            return super().generate(**generate_kwargs)

