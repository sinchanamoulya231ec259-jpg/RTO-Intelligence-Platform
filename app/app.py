import pandas as pd
from pathlib import Path
import joblib
import streamlit as st

st.set_page_config(
    page_title="RTO Intelligence Platform",
    page_icon="📦",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "rto_xgboost_model.pkl"


model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(BASE_DIR / "models" / "feature_columns.pkl")
explainer = joblib.load(BASE_DIR / "models" / "shap_explainer.pkl")
st.title("📦 AI RTO Risk Intelligence Platform")

st.markdown(
"""
Predict Return-to-Origin (RTO) risk before dispatch using Machine Learning.
"""
)

st.divider()

col1, col2 = st.columns([1,1])
with col1:

    st.subheader("👤 Customer Information")

    customer_tenure = st.number_input(
        "Customer Tenure (Days)",
        min_value=0,
        max_value=5000,
        value=365
    )

    total_orders = st.number_input(
        "Total Previous Orders",
        min_value=0,
        max_value=500,
        value=10
    )

    previous_rto = st.number_input(
        "Previous RTO Count",
        min_value=0,
        max_value=50,
        value=0
    )

    previous_return = st.number_input(
        "Previous Return Count",
        min_value=0,
        max_value=50,
        value=1
    )

    previous_cod_orders = st.number_input(
        "Previous COD Orders",
        min_value=0,
        max_value=100,
        value=5
    )

    previous_cod_refusal = st.number_input(
        "Previous COD Refusals",
        min_value=0,
        max_value=50,
        value=0
    )

    first_time_customer = st.selectbox(
        "First Time Customer",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    customer_region = st.selectbox(
        "Customer Region",
        [
            "Central",
            "East",
            "North",
            "Northeast",
            "South",
            "West"
        ]
    )

    payment_method = st.selectbox(
        "Payment Method",
        ["COD", "Prepaid"]
    )

    st.subheader("🏪 Seller Information")

    seller_rating = st.slider(
        "Seller Rating",
        1.0,
        5.0,
        4.2
    )

    seller_rto_rate = st.slider(
        "Seller RTO Rate (%)",
        0.0,
        100.0,
        10.0
    )

    average_dispatch_time = st.slider(
        "Average Dispatch Time (Days)",
        0.0,
        10.0,
        2.0
    )

    seller_cancellation_rate = st.slider(
        "Seller Cancellation Rate (%)",
        0.0,
        100.0,
        5.0
    )

    st.subheader("📦 Product Information")

    product_category = st.selectbox(
        "Product Category",
        [
            "Beauty & Personal Care",
            "Bedsheets & Furnishings",
            "Footwear",
            "Home Decor",
            "Jewellery & Accessories",
            "Kids Wear",
            "Kitchen & Home",
            "Men's T-Shirts",
            "Mobile Accessories",
            "Sarees",
            "Women Ethnic Wear"
        ]
    )

    brand = st.selectbox(
        "Brand",
        [
            "Comfy Nights",
            "Fabrino",
            "GlowNest",
            "Kalakriti",
            "Localoom",
            "Meraki Home",
            "Prakruti",
            "Rugged Roads",
            "Silverline Jewels",
            "StyleNest",
            "TrendKart Basics",
            "Urbane Threads",
            "Vastraa Villa",
            "Zorva"
        ]
    )

    order_value = st.number_input(
        "Order Value (₹)",
        min_value=100,
        max_value=5000,
        value=500
    )

    discount_percentage = st.slider(
        "Discount Percentage",
        0,
        90,
        20
    )

    quantity = st.slider(
        "Quantity",
        1,
        10,
        1
    )
    st.divider()

    predict = st.button(
       "🚀 Predict RTO Risk",
        use_container_width=True
    ) 

# ---------------- RIGHT ---------------- #

with col2:

    st.subheader("📊 Prediction Result")

    st.info("Prediction will appear here after clicking the button.")

    st.subheader("📈 Top Risk Factors")
    st.write("-")

    st.subheader("💼 Business Recommendation")
    st.write("-")




if predict:

    input_df = pd.DataFrame(
        0.0,
        index=[0],
        columns=feature_columns
    )

    # Numerical Features
    input_df.loc[0, "Customer_Tenure_Days"] = customer_tenure
    input_df.loc[0, "Total_Orders"] = total_orders
    input_df.loc[0, "Previous_RTO_Count"] = previous_rto
    input_df.loc[0, "Previous_Return_Count"] = previous_return
    input_df.loc[0, "Previous_COD_Orders"] = previous_cod_orders
    input_df.loc[0, "Previous_COD_Refusals"] = previous_cod_refusal
    input_df.loc[0, "First_Time_Customer"] = first_time_customer
    input_df.loc[0, "Seller_Rating"] = seller_rating
    input_df.loc[0, "Seller_RTO_Rate"] = seller_rto_rate
    input_df.loc[0, "Average_Dispatch_Time"] = average_dispatch_time
    input_df.loc[0, "Seller_Cancellation_Rate"] = seller_cancellation_rate
    input_df.loc[0, "Order_Value"] = order_value
    input_df.loc[0, "Discount_Percentage"] = discount_percentage
    input_df.loc[0, "Quantity"] = quantity

    # Default values for now
    input_df.loc[0, "Destination_City"] = 0
    input_df.loc[0, "Destination_PIN"] = 560001
    input_df.loc[0, "Estimated_Delivery_Days"] = 4
    input_df.loc[0, "Logistics_Cost"] = 80
        # One-hot encoded categorical features

    if payment_method == "Prepaid":
        input_df.loc[0, "Payment_Method_Prepaid"] = 1

    if customer_region != "Central":
        col = f"Customer_Region_{customer_region}"
        if col in input_df.columns:
            input_df.loc[0, col] = 1

    product_col = f"Product_Category_{product_category}"
    if product_col in input_df.columns:
        input_df.loc[0, product_col] = 1

    brand_col = f"Brand_{brand}"
    if brand_col in input_df.columns:
        input_df.loc[0, brand_col] = 1
    probability = model.predict_proba(input_df)[0][1]
    shap_values = explainer.shap_values(input_df)

    if probability < 0.30:
       risk = "🟢 LOW RISK"

    elif probability < 0.70:
       risk = "🟡 MEDIUM RISK"

    else:
       risk = "🔴 HIGH RISK"
 
    st.metric(
       label="Predicted RTO Probability",
       value=f"{probability:.1%}"
    )

    st.markdown(f"## {risk}")

    st.progress(int(probability * 100))
    

    
    st.subheader("📈 Top Risk Factors")

    importance = abs(shap_values[0])

    top_features = (
        pd.DataFrame({
            "Feature": input_df.columns,
            "Importance": importance
        })
        .sort_values("Importance", ascending=False)
        .head(5)
        
    )
    top_features = top_features.reset_index(drop=True)


    st.write("### 🚨 Key Risk Drivers")

    feature_mapping = {
        "Payment_Method_Prepaid": f"Payment Method: {payment_method}",
        "Previous_COD_Refusals": f"Customer has {previous_cod_refusal} previous COD refusals",
        "Previous_RTO_Count": f"Customer has {previous_rto} previous RTOs",
        "Total_Orders": f"Customer has placed {total_orders} previous orders",
        "Seller_Rating": f"Seller rating is {seller_rating:.1f}",
        "Seller_RTO_Rate": f"Seller historical RTO rate is {seller_rto_rate:.1f}%",
        "Estimated_Delivery_Days": "Estimated delivery is 4 days",
        "Logistics_Cost": "Logistics cost contributes to prediction",
        "Destination_City": "Destination city contributes to prediction",
        "Order_Value": f"Order value is ₹{order_value}"
    }

    for _, row in top_features.iterrows():
        feature = row["Feature"]

        if feature in feature_mapping:
            st.write(f"• {feature_mapping[feature]}")
        else:
            st.write(f"• {feature.replace('_', ' ')}")

    st.subheader("💼 Business Recommendations")

    recommendations = []

    if payment_method == "COD":
        recommendations.append("💰 Encourage prepaid payment to reduce RTO risk.")

    if previous_cod_refusal > 0:
        recommendations.append("📞 Call the customer to confirm the order before dispatch.")

    if previous_rto > 0:
        recommendations.append("📍 Verify the delivery address before shipping.")

    if seller_rating < 3.5:
        recommendations.append("⭐ Review seller performance and packaging quality.")

    if seller_rto_rate > 20:
        recommendations.append("📊 Monitor seller's historical RTO performance.")

    if order_value > 2000:
        recommendations.append("📦 Consider additional verification for high-value orders.")

    if len(recommendations) == 0:
        st.success("✅ No major business actions recommended. Proceed with normal dispatch.")
    else:
        for recommendation in recommendations:
            st.write(recommendation)