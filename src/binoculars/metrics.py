import numpy as np
import torch
import transformers

ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
softmax_fn = torch.nn.Softmax(dim=-1)


def perplexity(encoding: transformers.BatchEncoding,
               logits: torch.Tensor,
               median: bool = False,
               temperature: float = 1.0):
    shifted_logits = logits[..., :-1, :].contiguous() / temperature
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

    if median:
        ce_nan = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels).
                  masked_fill(~shifted_attention_mask.bool(), float("nan")))
        ppl = np.nanmedian(ce_nan.cpu().float().numpy(), 1)

    else:
        ppl = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels) *
               shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
        ppl = ppl.to("cpu").float().numpy()

    return ppl


def entropy(p_logits: torch.Tensor,
            q_logits: torch.Tensor,
            encoding: transformers.BatchEncoding,
            pad_token_id: int,
            median: bool = False,
            sample_p: bool = False,
            temperature: float = 1.0):
    vocab_size = p_logits.shape[-1]
    total_tokens_available = q_logits.shape[-2]
    p_scores, q_scores = p_logits / temperature, q_logits / temperature

    p_proba = softmax_fn(p_scores).view(-1, vocab_size)

    if sample_p:
        p_proba = torch.multinomial(p_proba.view(-1, vocab_size), replacement=True, num_samples=1).view(-1)

    q_scores = q_scores.view(-1, vocab_size)

    ce = ce_loss_fn(input=q_scores, target=p_proba).view(-1, total_tokens_available)
    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)

    if median:
        ce_nan = ce.masked_fill(~padding_mask.bool(), float("nan"))
        agg_ce = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        agg_ce = (((ce * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy())

    return agg_ce


def kl_divergence(p_logits: torch.Tensor,
                  q_logits: torch.Tensor,
                  encoding: transformers.BatchEncoding,
                  temperature: float = 1.0):
    """
    Compute KL Divergence (P || Q) averaged over valid tokens.
    """
    # Shift logits to align with next-token prediction positions (1 to N)
    p_logits = p_logits[..., :-1, :].contiguous() / temperature
    q_logits = q_logits[..., :-1, :].contiguous() / temperature
    
    # Compute probabilities
    p_probs = softmax_fn(p_logits)
    p_log_probs = torch.nn.functional.log_softmax(p_logits, dim=-1)
    q_log_probs = torch.nn.functional.log_softmax(q_logits, dim=-1)
    
    # KL(P || Q) = sum(P * (log P - log Q))
    kl_per_token = torch.sum(p_probs * (p_log_probs - q_log_probs), dim=-1)
    
    # Masking
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()
    
    # Average
    kl_mean = (kl_per_token * shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
    return kl_mean.to("cpu").float().numpy()


def js_divergence(p_logits: torch.Tensor,
                  q_logits: torch.Tensor,
                  encoding: transformers.BatchEncoding,
                  temperature: float = 1.0):
    """
    Compute Jensen-Shannon Divergence averaged over valid tokens.
    JSD(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q)
    """
    p_logits = p_logits[..., :-1, :].contiguous() / temperature
    q_logits = q_logits[..., :-1, :].contiguous() / temperature
    
    p_probs = softmax_fn(p_logits)
    q_probs = softmax_fn(q_logits)
    
    m_probs = 0.5 * (p_probs + q_probs)
    
    # We need log probabilities for KL
    p_log_probs = torch.nn.functional.log_softmax(p_logits, dim=-1)
    q_log_probs = torch.nn.functional.log_softmax(q_logits, dim=-1)
    m_log_probs = torch.log(m_probs + 1e-9) # Add epsilon for stability
    
    # KL(P || M)
    kl_p_m = torch.sum(p_probs * (p_log_probs - m_log_probs), dim=-1)
    
    # KL(Q || M)
    kl_q_m = torch.sum(q_probs * (q_log_probs - m_log_probs), dim=-1)
    
    jsd_per_token = 0.5 * (kl_p_m + kl_q_m)
    
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()
    jsd_mean = (jsd_per_token * shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
    
    return jsd_mean.to("cpu").float().numpy()
