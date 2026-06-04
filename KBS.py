# =====================================================
# KNOWLEDGE BASED SYSTEM
# Student Performance Prediction Project
# =====================================================

# -------------------------
# KNOWLEDGE BASE
# -------------------------

rules = {

    "R1": {
        "if": ["low_attendance", "low_motivation"],
        "then": "high_academic_risk"
    },

    "R2": {
        "if": ["high_study_hours", "high_motivation"],
        "then": "strong_study_habits"
    },

    "R3": {
        "if": ["strong_study_habits"],
        "then": "low_academic_risk"
    },

    "R4": {
        "if": ["high_previous_scores"],
        "then": "strong_academic_background"
    },

    "R5": {
        "if": [
            "strong_academic_background",
            "strong_study_habits"
        ],
        "then": "high_success_chance"
    },

    "R6": {
        "if": ["low_resources"],
        "then": "resource_deficiency"
    },

    "R7": {
        "if": ["resource_deficiency"],
        "then": "recommend_resources"
    },

    "R8": {
        "if": [
            "high_teacher_quality",
            "high_attendance"
        ],
        "then": "excellent_learning_environment"
    },

    "R9": {
        "if": ["negative_peer_influence"],
        "then": "social_risk"
    },

    "R10": {
        "if": [
            "social_risk",
            "high_academic_risk"
        ],
        "then": "counseling_required"
    },

    "R11": {
        "if": [
            "ml_poor",
            "high_academic_risk"
        ],
        "then": "at_risk_student"
    },

    "R12": {
        "if": [
            "ml_good",
            "high_success_chance"
        ],
        "then": "high_performer"
    }
}


# -------------------------
# FORWARD CHAINING
# -------------------------

def forward_chaining(initial_facts):

    facts = set(initial_facts)

    trace = []

    changed = True

    while changed:

        changed = False

        for rule_name, rule in rules.items():

            if all(
                condition in facts
                for condition in rule["if"]
            ):

                if rule["then"] not in facts:

                    facts.add(rule["then"])

                    trace.append(
                        f"{rule_name} -> {rule['then']}"
                    )

                    changed = True

    return facts, trace


# -------------------------
# BACKWARD CHAINING
# -------------------------

def backward_chaining(goal, facts):

    if goal in facts:
        return True

    for rule in rules.values():

        if rule["then"] == goal:

            return all(
                backward_chaining(
                    condition,
                    facts
                )
                for condition in rule["if"]
            )

    return False
