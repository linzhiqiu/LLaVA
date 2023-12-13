#    Copyright 2023 Zhiqiu Lin
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForSeq2SeqLM

from transformers.modeling_outputs import Seq2SeqLMOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForSeq2SeqLM
from transformers import T5Config, T5ForConditionalGeneration


class T5LlavaConfig(T5Config):
    model_type = "t5_llava"


class LlavaT5ForConditionalGeneration(LlavaMetaModel, LlavaMetaForSeq2SeqLM, T5ForConditionalGeneration): # To make multiple inheritance work, need to put T5ForConditionalGeneration at the end
    # This class supports both T5 and FlanT5
    config_class = T5LlavaConfig

    def __init__(self, config):
        super(LlavaT5ForConditionalGeneration, self).__init__(config)
        self.embed_tokens = self.encoder.embed_tokens

    def get_model(self):
        return self

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        _, attention_mask, decoder_attention_mask, past_key_values, inputs_embeds, labels = \
            self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, decoder_attention_mask, past_key_values, labels, images)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = super(LlavaT5ForConditionalGeneration, self).forward(
            input_ids=None, # will be None if inputs_embeds is not None
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        return outputs

    # def prepare_inputs_for_generation(
    #     self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    # ):
    #     if past_key_values:
    #         input_ids = input_ids[:, -1:]

    #     # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    #     if inputs_embeds is not None and past_key_values is None:
    #         model_inputs = {"inputs_embeds": inputs_embeds}
    #     else:
    #         model_inputs = {"input_ids": input_ids}

    #     model_inputs.update(
    #         {
    #             "past_key_values": past_key_values,
    #             "use_cache": kwargs.get("use_cache"),
    #             "attention_mask": attention_mask,
    #             "images": kwargs.get("images", None),
    #         }
    #     )
    #     return model_inputs

AutoConfig.register("t5_llava", T5LlavaConfig)
AutoModelForSeq2SeqLM.register(T5LlavaConfig, LlavaT5ForConditionalGeneration)
