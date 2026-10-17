"""
VIPER Policy Distillation (`sklearn.tree.DecisionTreeClassifier`) for Pokemon Auto Chess.

Implements Bastani et al. (NeurIPS 2018) "Verifiable Reinforcement Learning via Policy
Extraction" (VIPER):
- Collects on-policy trajectories from trained PyTorch `RewardConditionedMaskedActorCritic`
  agents across the Community League.
- Computes importance weights $w(s) = \max_a \log \pi(a|s) - \mathbb{E}_{a'}[\log \pi(a'|s)]$
  so critical states (e.g. low-HP all-in, 50g interest cap, 3-of-a-kind roll) dominate
  the tree splits.
- Fits shallow, human-interpretable `DecisionTreeClassifier(max_depth=4)` models per
  archetype and exports both `export_text` trees, feature importance rankings, and
  explicit `SE ... ENTÃO ...` Portuguese heuristic rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
import torch

from .gym_env import (
    ACT_BUY_0,
    ACT_BUY_5,
    ACT_DEPLOY_0,
    ACT_DEPLOY_5,
    ACT_END_TURN,
    ACT_LEVEL_UP,
    ACT_LOCK_SHOP,
    ACT_REROLL,
    ACT_SELL_0,
    ACT_SELL_5,
    PokemonAutoChessGymEnv,
    RewardWeightConfig,
)
from .models import RewardConditionedMaskedActorCritic


MACRO_ACTION_NAMES = [
    "COMPRAR_UNIDADE_SINERGIA_OU_PAR",
    "REROLL_LOJA_1G",
    "COMPRAR_XP_SUBIR_NIVEL_4G",
    "OTIMIZAR_OU_VENDER_BANCO",
    "GUARDAR_OURO_ENCERRAR_TURNO",
]

FEATURE_NAMES = [
    "stage_level",
    "player_hp",
    "gold",
    "interest_tier",
    "player_level",
    "xp_to_next_level",
    "win_streak",
    "lose_streak",
    "board_size",
    "free_board_slots",
    "bench_size",
    "relative_board_power",
    "shop_pair_matches",
    "shop_synergy_matches",
    "num_2star_units",
    "num_3star_units",
    "max_synergy_tier",
    "alive_opponents",
]


def map_to_macro_action(action: int) -> int:
    if ACT_BUY_0 <= action <= ACT_BUY_5:
        return 0  # COMPRAR_UNIDADE_SINERGIA_OU_PAR
    elif action == ACT_REROLL:
        return 1  # REROLL_LOJA_1G
    elif action == ACT_LEVEL_UP:
        return 2  # COMPRAR_XP_SUBIR_NIVEL_4G
    elif (ACT_DEPLOY_0 <= action <= ACT_DEPLOY_5) or (ACT_SELL_0 <= action <= ACT_SELL_5):
        return 3  # OTIMIZAR_OU_VENDER_BANCO
    else:
        return 4  # GUARDAR_OURO_ENCERRAR_TURNO


@dataclass
class DistilledHeuristicReport:
    agent_name: str
    archetype: str
    fidelity_accuracy: float
    tree_text: str
    top_features: List[Tuple[str, float]]
    extracted_rules_pt: List[str]
    action_distribution: Dict[str, float]


class VIPERPolicyDistiller:
    """
    Distills a trained PyTorch `RewardConditionedMaskedActorCritic` policy into a
    verifiable `scikit-learn` Decision Tree using Q-advantage sample weighting.
    """

    def __init__(self, max_depth: int = 4, min_samples_leaf: int = 12):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def distill_agent(
        self,
        env: PokemonAutoChessGymEnv,
        model: RewardConditionedMaskedActorCritic,
        config: RewardWeightConfig,
        num_episodes: int = 16,
        seed: int = 100,
    ) -> DistilledHeuristicReport:
        env.set_reward_config(config)
        model.eval()

        X_rows: List[List[float]] = []
        y_labels: List[int] = []
        sample_weights: List[float] = []

        for ep in range(num_episodes):
            obs, _ = env.reset(seed=seed + ep * 17)
            done = False
            while not done:
                feats_dict = env.extract_interpretable_features()
                x_vec = [feats_dict[k] for k in FEATURE_NAMES]
                mask = env.action_masks()

                with torch.no_grad():
                    t_obs = torch.from_numpy(obs).unsqueeze(0)
                    t_mask = torch.from_numpy(mask).unsqueeze(0)
                    h = model.forward_trunk(t_obs)
                    logits = model.actor_head(h) + model.compute_Domain_prior_logits(t_obs)
                    masked_logits = logits.masked_fill(~t_mask, -1e8).squeeze(0)
                    probs = torch.softmax(masked_logits, dim=-1).cpu().numpy()
                    best_action = int(np.argmax(probs))

                    # VIPER criticality weight: gap between best action log-prob and average valid log-prob
                    valid_logits = masked_logits[t_mask.squeeze(0)].cpu().numpy()
                    weight = float(np.max(valid_logits) - np.mean(valid_logits) + 0.1)

                macro_act = map_to_macro_action(best_action)
                X_rows.append(x_vec)
                y_labels.append(macro_act)
                sample_weights.append(max(0.05, weight))

                obs, _, terminated, truncated, _ = env.step(best_action)
                done = terminated or truncated

        X = np.array(X_rows, dtype=np.float32)
        y = np.array(y_labels, dtype=np.int64)
        w = np.array(sample_weights, dtype=np.float32)

        clf = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
        )
        clf.fit(X, y, sample_weight=w)

        preds = clf.predict(X)
        fidelity = float(np.mean(preds == y))

        # Present classes in this agent's trajectory
        present_classes = sorted(list(set(y)))
        class_names = [MACRO_ACTION_NAMES[c] for c in present_classes]
        tree_str = export_text(
            clf,
            feature_names=FEATURE_NAMES,
            class_names=class_names,
            decimals=2,
        )

        importances = clf.feature_importances_
        ranked_feats = sorted(
            zip(FEATURE_NAMES, [float(v) for v in importances]),
            key=lambda x: x[1],
            reverse=True,
        )[:6]

        # Action distribution
        act_dist: Dict[str, float] = {}
        for idx, name in enumerate(MACRO_ACTION_NAMES):
            act_dist[name] = float(np.mean(y == idx))

        rules_pt = self._extract_readable_rules_from_tree(clf, X, y)

        return DistilledHeuristicReport(
            agent_name=config.name,
            archetype=config.archetype,
            fidelity_accuracy=fidelity,
            tree_text=tree_str,
            top_features=ranked_feats,
            extracted_rules_pt=rules_pt,
            action_distribution=act_dist,
        )

    def _extract_readable_rules_from_tree(
        self,
        clf: DecisionTreeClassifier,
        X: np.ndarray,
        y: np.ndarray,
    ) -> List[str]:
        """
        Walks the root-to-leaf paths of the fitted `DecisionTreeClassifier` and formats
        the highest-coverage paths into crisp `SE ... ENTÃO ...` rules in Portuguese.
        """
        tree_ = clf.tree_
        feature_name = [
            FEATURE_NAMES[i] if i != -2 else "leaf"
            for i in tree_.feature
        ]

        paths: List[Tuple[int, str]] = []

        def recurse(node: int, conds: List[str]):
            if tree_.feature[node] != -2:
                name = feature_name[node]
                th = tree_.threshold[node]
                recurse(tree_.children_left[node], conds + [f"{name} <= {th:.2f}"])
                recurse(tree_.children_right[node], conds + [f"{name} > {th:.2f}"])
            else:
                counts = tree_.value[node][0]
                best_cls_idx = int(np.argmax(counts))
                actual_cls = int(clf.classes_[best_cls_idx])
                n_samples = int(tree_.n_node_samples[node])
                purity = float(counts[best_cls_idx] / max(1e-6, np.sum(counts)))
                cond_str = " E ".join(conds) if conds else "SEMPRE"
                rule = (
                    f"SE ({cond_str}) ENTÃO [{MACRO_ACTION_NAMES[actual_cls]}] "
                    f"(confiança={purity*100:.1f}%, n={n_samples})"
                )
                paths.append((n_samples, rule))

        recurse(0, [])
        paths.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in paths[:5]]
