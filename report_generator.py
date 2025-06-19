from fpdf import FPDF
import os

def generate_anomaly_report(df, output_path="output/anomaly_report.pdf"):
    anomalies = df[df['is_anomaly'] == True].copy()
    if anomalies.empty:
        return None

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Anomaly Detection Report", ln=True, align='C')
    pdf.set_font("Arial", size=10)

    for _, row in anomalies.iterrows():
        pdf.ln(5)
        emp_id = row.get("EmployeeID", "Unknown")
        rank = row.get("Rank", "Unknown")
        base = row.get("BasePay", "N/A")
        bonus = row.get("Bonus", "N/A")
        yos = row.get("YearsOfService", "N/A")
        prom_rate = row.get("PromotionRate", "N/A")
        reason_parts = []

        if row.get("BonusRatio", 0) > 0.5:
            reason_parts.append("High bonus-to-base-pay ratio")
        if row.get("PromotionRate", 0) > 2:
            reason_parts.append("Rapid promotion rate")
        if row.get("BasePayPerYear", 0) > 10000:
            reason_parts.append("High pay per year of service")

        reason_text = "; ".join(reason_parts) or "Unusual data pattern detected."

        pdf.multi_cell(0, 10, f"EmployeeID: {emp_id}\nRank: {rank}\nYears of Service: {yos}\nBasePay: {base}\nBonus: {bonus}\nReason: {reason_text}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pdf.output(output_path)
    return output_path
