"""
Custom reward function for verl PPO training.

Sends each generated response to the reward model server (reward_server.py)
and returns a normalized score. The server must be running before training starts.
"""

import requests
import wandb
from typing import Optional

SERVER_URL = "http://127.0.0.1:8000/score"

_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=30,
            max_retries=3,
            pool_block=False,
        )
        _session.mount("http://", adapter)
        _session.mount("https://", adapter)
    return _session


def compute_score(data_source, solution_str, ground_truth, extra_info):
    """
    Reward function called by verl's RewardManager.

    Args:
        data_source:  Dataset identifier (unused).
        solution_str: The generated response text.
        ground_truth: Expected answer (unused for generative reward).
        extra_info:   Dict containing 'query' and 'profile'.

    Returns:
        Float reward score.
    """
    payload = {
        "response_text": solution_str,
        "query": extra_info.get("query", ""),
        "profile": extra_info.get("profile", ""),
    }

    try:
        response = _get_session().post(SERVER_URL, json=payload, timeout=60)
        response.raise_for_status()
        score = response.json()["score"] / 10
    except requests.exceptions.Timeout:
        print(f"[Reward] Request timeout — returning default score")
        score = 0.5
    except Exception as e:
        print(f"[Reward] Error: {e} — returning default score")
        score = 0.5

    if wandb.run is not None:
        wandb.log({"reward/score": score}, commit=False)

    return score


def cleanup():
    """Close the HTTP session. Call on shutdown."""
    global _session
    if _session is not None:
        _session.close()
        _session = None
