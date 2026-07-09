import ltn
import torch
import torch.nn as nn


class HypotensionPredicate(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, sbp):
        sbp = torch.nan_to_num(sbp, nan=120.0, posinf=250.0, neginf=50.0)
        out = torch.sigmoid(-(sbp - 90.0) / 5.0)
        return torch.clamp(out, 0.0, 1.0)


class HypoxiaPredicate(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, spo2):
        spo2 = torch.nan_to_num(spo2, nan=98.0, posinf=100.0, neginf=50.0)
        out = torch.sigmoid(-(spo2 - 90.0) / 3.0)
        return torch.clamp(out, 0.0, 1.0)


class SevereShockPredicate(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, shock_index):
        shock_index = torch.nan_to_num(shock_index, nan=0.5, posinf=3.0, neginf=0.1)
        shock_index = torch.clamp(shock_index, 0.1, 3.0)
        out = torch.sigmoid((shock_index - 1.0) / 0.2)
        return torch.clamp(out, 0.0, 1.0)


class PhysiologicalBoundaryPredicate(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, vitals):
        """
        vitals: [batch, features=5]
        Features: [HR, SBP, DBP, SpO2, Shock Index]
        """
        vitals = torch.nan_to_num(vitals, nan=0.0, posinf=1e3, neginf=-1e3)
        
        hr = vitals[:, 0]
        sbp = vitals[:, 1]
        dbp = vitals[:, 2]
        spo2 = vitals[:, 3]
        shock_index = vitals[:, 4]
        
        shock_index = torch.clamp(shock_index, 0.1, 3.0)
        
        hr_valid = torch.sigmoid((hr - 20) / 10) * torch.sigmoid(-(hr - 200) / 10)
        sbp_valid = torch.sigmoid((sbp - 50) / 10) * torch.sigmoid(-(sbp - 250) / 10)
        dbp_valid = torch.sigmoid((dbp - 30) / 10) * torch.sigmoid(-(dbp - 150) / 10)
        spo2_valid = torch.sigmoid((spo2 - 50) / 10) * torch.sigmoid(-(spo2 - 100.5) / 2)
        si_valid = torch.sigmoid((shock_index - 0.1) / 0.1) * torch.sigmoid(-(shock_index - 3.0) / 0.5)
        
        validity_scores = torch.stack([hr_valid, sbp_valid, dbp_valid, spo2_valid, si_valid], dim=-1)
        total = validity_scores.sum(dim=-1)
        out = total - 5 + 1
        return torch.clamp(out, min=0.0, max=1.0)


class SepsisKnowledgeBase:
    def __init__(self):
        self.Pred_Hypotension = ltn.Predicate(HypotensionPredicate())
        self.Pred_Hypoxia = ltn.Predicate(HypoxiaPredicate())
        self.Pred_SevereShock = ltn.Predicate(SevereShockPredicate())
        self.Pred_Physiological = ltn.Predicate(PhysiologicalBoundaryPredicate())
        
        self.And = ltn.Connective(ltn.fuzzy_ops.AndLuk())
        self.Or = ltn.Connective(ltn.fuzzy_ops.OrLuk())
        self.Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesLuk())
        self.Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
        self.Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
    
    def compute_satisfiability(self, vitals, sepsis_label):
        """
        vitals: [batch, seq=24, features=5]
        sepsis_label: [batch]
        """
        batch_size, seq_len, n_features = vitals.shape
        vitals = torch.nan_to_num(vitals, nan=0.0, posinf=1e3, neginf=-1e3)
        
        # Extract: SBP=1, SpO2=3, ShockIndex=4
        sbp = vitals[:, :, 1]
        spo2 = vitals[:, :, 3]
        shock_index = vitals[:, :, 4]
        
        # Compute dysfunction
        hypo = self.Pred_Hypotension.model(sbp)
        hypox = self.Pred_Hypoxia.model(spo2)
        shock = self.Pred_SevereShock.model(shock_index)
        
        any_dysfunction_hourly = torch.max(hypo, torch.max(hypox, shock))
        dysfunction_per_patient = any_dysfunction_hourly.mean(dim=1)
        
        # Physiological boundaries
        vitals_flat = vitals.reshape(-1, n_features)
        vitals_var = ltn.Variable("v", vitals_flat)
        axiom_validity = self.Forall(vitals_var, self.Pred_Physiological(vitals_var))
        
        # Sepsis → Dysfunction
        axiom_sepsis_val = torch.mean(
            torch.clamp(1.0 - sepsis_label.float() + dysfunction_per_patient, 0.0, 1.0)
        )
        
        kb_satisfiability = torch.clamp(axiom_validity.value + axiom_sepsis_val - 1.0, 0.0, 1.0)
        return kb_satisfiability


def logical_loss(vitals, sepsis_label, knowledge_base):
    satisfiability = knowledge_base.compute_satisfiability(vitals, sepsis_label)
    loss = 1.0 - satisfiability
    return torch.clamp(loss, 0.0, 1.0)