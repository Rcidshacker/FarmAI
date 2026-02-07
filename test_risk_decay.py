from datetime import datetime, timedelta

def calculate_protection_efficacy(last_spray_date, current_date):
    """
    Simplified extract of the logic from src/api/routes/pest.py
    """
    days_passed = (current_date.date() - last_spray_date.date()).days
    
    print(f"   [Day {days_passed}]", end=" ")
    
    if days_passed < 0: return 0.0
    if days_passed > 14: return 0.0
    
    # Base effective days
    effective_days = float(days_passed)
    
    # Curve Logic
    if effective_days < 2.0:
        efficacy = 0.5 + (0.25 * effective_days) 
        phase = "Ramp-up (Spreading)"
    elif effective_days < 5.0:
        efficacy = 1.0
        phase = "Plateau (Max Protection)"
    else:
        remaining_days = 14.0 - 5.0 # 9 days
        decay_progress = (effective_days - 5.0) / remaining_days
        efficacy = (1.0 - decay_progress) ** 2
        phase = "Decay (Wearing off)"
    
    final_efficacy = max(0.0, min(1.0, efficacy))
    
    # Logic from pest.py: risk_reduction_mult = 1.0 - (protection_factor * 0.90)
    risk_reduction = final_efficacy * 0.90
    remaining_risk_factor = 1.0 - risk_reduction
    
    print(f"Phase: {phase:<25} | Protection: {final_efficacy*100:>5.1f}% | Risk Reduced by: {risk_reduction*100:>5.1f}% | Net Risk Factor: {remaining_risk_factor:.2f}")

def run_simulation():
    print("=== Simulating 'Mark as Sprayed' Risk Curve ===")
    print("Assumption: User sprayed on Day 0.")
    print("Baseline Risk: 100% (1.0) -> See how it reduces over time.\n")
    
    spray_date = datetime.now()
    
    for i in range(16):
        check_date = spray_date + timedelta(days=i)
        calculate_protection_efficacy(spray_date, check_date)

if __name__ == "__main__":
    run_simulation()
