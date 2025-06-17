import streamlit as st
from pages.session_config.lang_detail import get_translation
import base64

st.set_page_config(
    page_title="Detail Implementasi",
    page_icon=":bulb:",
    layout="wide"
)

st.markdown("""
    <style>
        body {
        background-color: #FFFFFF;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)

def get_base64_image(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
    
def get_base64_image_datauri(path):
    ext = path.split('.')[-1]
    mime = f"image/{ext if ext != 'jpg' else 'jpeg'}"
    with open(path, "rb") as img_file:
        base64_str = base64.b64encode(img_file.read()).decode()
        return f"data:{mime};base64,{base64_str}"

logo1 = get_base64_image("./images/logo_kampus.png")
logo2 = get_base64_image("./images/logo_kampus_merdeka.png")
logo3 = get_base64_image("./images/logo_kemendikbud.png")
st.markdown(f"""
    <style>
        .logo-container {{
            position: absolute;
            top: 10px;
            right: 20px;
            display: flex;
            gap: 10px;
        }}
        .logo-container img {{
            height: 50px;
        }}
    </style>
    <div class="logo-container">
        <img src="data:image/png;base64,{logo1}" />
        <img src="data:image/png;base64,{logo2}" />
        <img src="data:image/png;base64,{logo3}" />
    </div>
""", unsafe_allow_html=True)


st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)


st.markdown(
    r"""
    <style>
    .stAppDeployButton {
            visibility: hidden;
        }
    </style>
    """, unsafe_allow_html=True
)
st.markdown(
    """<style> .reportview-container { margin-top: -2em; } #MainMenu {visibility: hidden;} .stDeployButton {display:none;} footer {visibility: hidden;} #stDecoration {display:none;} </style>""", unsafe_allow_html=True)


if 'selected_language' not in st.session_state:
    st.session_state['selected_language'] = 'id'
st.title(get_translation(st.session_state['selected_language'], 'page_title'))

# Tentang Halaman
st.markdown(f"""<div style=" padding: 15px; border-left: 6px solid #2c6faf; border-radius: 8px; font-size: 16px;">
    <strong>{get_translation(st.session_state['selected_language'], 'about_this_page_label')}</strong><br>
    {get_translation(st.session_state['selected_language'], 'about_this_page')}</div>""", unsafe_allow_html=True)

# Teknologi yang digunakan
st.markdown(f"""<div style="text-align: center; padding-top: 10px;">
    <h4 style="margin-bottom: 20px;">🛠️ {get_translation(st.session_state['selected_language'], 'tech_stack')}</h4>
</div>""", unsafe_allow_html=True)

# Gambar Python dan Yahoo Finance
img_python_logo = './images/python-logo.png'
img_yahoo_finance = './images/yahoo-finance_BIG.png'

col1, col2 = st.columns([1, 1])
with col1:
    st.image(get_base64_image_datauri(img_python_logo), width="auto",
             caption="Python", use_column_width="auto")
with col2:
    st.image(get_base64_image_datauri(img_yahoo_finance), width="auto",
             caption="Yahoo Finance", use_column_width="auto")

st.markdown(f"""<div style="padding: 10px 0;">
    <h4 style='margin-bottom: 5px;'>{get_translation(st.session_state['selected_language'], 'time_range')}</h4>
    <p style='margin-top: 0;'><strong>{get_translation(st.session_state['selected_language'], 'time_used')}</strong></p>
</div>""", unsafe_allow_html=True)

st.markdown(f"""<div style="background-color: #e6f0fa; padding: 15px; border-left: 6px solid #2c6faf; border-radius: 8px; font-size: 16px;">
    <strong>{get_translation(st.session_state['selected_language'], 'data_used')}</strong><br>
    {get_translation(st.session_state['selected_language'], 'data_used_description')}</div>
""", unsafe_allow_html=True)


companies = {
    "TLKM": "PT Telkom Indonesia Tbk",
    "ISAT": "Indosat Tbk PT",
    "EXCL": "XL Axiata Tbk PT"
}

company_info = f"""
<div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px; margin-top: 20px;">
    <h4 style="margin-bottom: 5px;">{get_translation(st.session_state['selected_language'], 'companies_label')}</h4>
    <ul style="padding-left: 20px;">
"""

for ticker, name in companies.items():
    company_info += f"<li><strong>{ticker}</strong> - {name}</li>"

company_info += f"""</ul>
    <p style="margin-top: 15px;">
       {get_translation(st.session_state['selected_language'], 'companies_available_description')}
    </p>

</div>"""

st.markdown(company_info, unsafe_allow_html=True)


st.markdown(f"""
<div style="background-color: #f9f9f9; padding: 20px 30px; border-radius: 12px; margin-top: 20px; margin-bottom:20px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);">
    <h4 style='text-align: center; color: #333333;'>🔍 Train/Test Split</h4>
    <p style='text-align: justify; font-size: 15px; color: #444444; margin-top: 10px;'>
        {get_translation(st.session_state['selected_language'], 'train_test_split_description')}
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f"<div style='text-align: center; font-weight: bold;'>{get_translation(st.session_state['selected_language'], 'data_train_label')}</div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; font-size: 28px; color: green;'>80%</div>",
                unsafe_allow_html=True)

with col2:
    st.markdown(
        f"<div style='text-align: center; font-weight: bold;'>{get_translation(st.session_state['selected_language'], 'data_test_label')}</div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; font-size: 28px; color: orange;'>20%</div>",
                unsafe_allow_html=True)

st.write(
    f"### {get_translation(st.session_state['selected_language'], 'lstm_detail_label')}")


img_lstm_flow = "./images/lstm-flow.jpg"
st.image(get_base64_image_datauri(img_lstm_flow), caption="LSTM Flow Architecture",
         use_column_width="always")

st.markdown(f"""
<div style="
    background-color: #e6f0fa;
    padding: 15px;
    border-left: 6px solid #2c6faf;
    border-radius: 8px;
    font-size: 16px;
">
    <strong>{get_translation(st.session_state['selected_language'], 'how_lstm_works_label')}</strong><br>
    {get_translation(st.session_state['selected_language'], 'lstm_description')}
</div>
""", unsafe_allow_html=True)

img_lstm_cell = "./images/lstm-cell.jpg"
st.image(get_base64_image_datauri(img_lstm_cell), caption="LSTM Cell Architecture",
         use_column_width="always")

st.markdown(f"""
<div style="
    background-color: #e6f0fa;
    padding: 15px;
    border-left: 6px solid #2c6faf;
    border-radius: 8px;
    font-size: 16px;
">
    <strong>{get_translation(st.session_state['selected_language'], 'inside_of_lstm_label')}</strong><br>
   {get_translation(st.session_state['selected_language'], 'inside_cell_lstm_description')}
</div>
""", unsafe_allow_html=True)


st.markdown(f"""
---

### 🔍 {get_translation(st.session_state['selected_language'], 'loss_curve_label')}
{get_translation(st.session_state['selected_language'], 'loss_curve_description')}
> *"{get_translation(st.session_state['selected_language'], 'loss_curve_highlight')}"*
""")

loss_plot = "./images/loss-plot.png"
st.image(get_base64_image_datauri(loss_plot), caption="Loss result", use_column_width=True)


st.markdown(f"""
### 📈 {get_translation(st.session_state['selected_language'], 'model_eval_label')}

{get_translation(st.session_state['selected_language'], 'model_eval_description')}

---

#### 🔹 1. Root Mean Squared Error (RMSE)

{get_translation(st.session_state['selected_language'], 'rmse_description')}

{get_translation(st.session_state['selected_language'], 'rmse_description_2')}
""")

# Display local RMSE formula image
rmse_formula = "./images/rmse.png"
st.image(get_base64_image_datauri(rmse_formula), caption="RMSE Formula", use_column_width=False)

st.markdown(f"""
---

#### 🔹 2. Mean Absolute Percentage Error (MAPE)
            
{get_translation(st.session_state['selected_language'], 'mape_description')}

{get_translation(st.session_state['selected_language'], 'mape_description_2')}
""")

# Display local MAPE formula image
mape_formula = "./images/mape.png"
st.image(get_base64_image_datauri(mape_formula), caption="MAPE Formula", use_column_width=False)


# Footer
st.markdown(f"""
    <style>
    .footer {{
        bottom: 0;
        width: 100%;
        background-color: #f0f2f6;
        color: #666;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        border-top: 1px solid #ccc;
    }}
    </style>
    <div class="footer">
        <b>{get_translation(st.session_state['selected_language'], 'copyright')}</b>
    </div>
""", unsafe_allow_html=True)
