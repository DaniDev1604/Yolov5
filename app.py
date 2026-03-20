from PIL import Image
import io
import streamlit as st
import numpy as np
import pandas as pd
import torch

st.set_page_config(
    page_title="¿Qué estoy viendo?",
    page_icon="🔍",
    layout="wide"
)

# 🎨 ESTILOS PERSONALIZADOS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

h1, h2, h3 {
    color: #38bdf8;
}

section[data-testid="stSidebar"] {
    background-color: #020617;
    border-right: 1px solid #1e293b;
}

.block-container {
    padding: 2rem;
}

</style>
""", unsafe_allow_html=True)

# 🤖 CARGA DEL MODELO
@st.cache_resource
def load_model():
    try:
        from ultralytics import YOLO
        model = YOLO("yolov5su.pt")
        return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        return None

# 🧠 HEADER
st.markdown("""
<h1 style='text-align: center;'>🔍 ¿Qué estoy viendo?</h1>
<p style='text-align: center; font-size:18px;'>
Captura una imagen y detecto objetos en tiempo real con IA
</p>
""", unsafe_allow_html=True)

with st.spinner("🚀 Cargando modelo YOLO..."):
    model = load_model()

# ⚙️ SIDEBAR
if model:
    with st.sidebar:
        st.markdown("## ⚙️ Configuración")
        st.markdown("---")

        conf_threshold = st.slider("🎯 Confianza mínima", 0.0, 1.0, 0.25, 0.01)
        iou_threshold  = st.slider("📦 Umbral IoU", 0.0, 1.0, 0.45, 0.01)
        max_det        = st.number_input("🔢 Máx detecciones", 10, 2000, 1000, 10)

        st.markdown("---")
        st.info("💡 Ajusta los parámetros para mejorar la detección.")

    # 📸 CAPTURA
    with st.container():
        st.markdown("### 📸 Captura tu imagen")
        picture = st.camera_input("Toma una foto", key="camera")

    if picture:
        bytes_data = picture.getvalue()

        pil_img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
        np_img  = np.array(pil_img)[..., ::-1]

        with st.spinner("🧠 Analizando imagen con IA..."):
            try:
                results = model(
                    np_img,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    max_det=int(max_det)
                )
            except Exception as e:
                st.error(f"Error durante la detección: {str(e)}")
                st.stop()

        st.success("✅ Detección completada")

        result    = results[0]
        boxes     = result.boxes
        annotated = result.plot()
        annotated_rgb = annotated[:, :, ::-1]

        col1, col2 = st.columns(2)

        # 🖼️ IMAGEN
        with col1:
            st.markdown("### 🖼️ Resultado")
            st.image(annotated_rgb, use_container_width=True)

        # 📊 RESULTADOS
        with col2:
            st.markdown("### 📊 Objetos detectados")

            if boxes is not None and len(boxes) > 0:
                label_names    = model.names
                category_count = {}
                category_conf  = {}

                for box in boxes:
                    cat  = int(box.cls.item())
                    conf = float(box.conf.item())

                    category_count[cat] = category_count.get(cat, 0) + 1
                    category_conf.setdefault(cat, []).append(conf)

                data = [
                    {
                        "Categoría": label_names[cat],
                        "Cantidad": count,
                        "Confianza promedio": f"{np.mean(category_conf[cat]):.2f}"
                    }
                    for cat, count in category_count.items()
                ]

                df = pd.DataFrame(data)

                # 🎯 TARJETAS VISUALES
                for row in data:
                    st.markdown(f"""
                    <div style="
                        background-color:#1e293b;
                        padding:15px;
                        border-radius:10px;
                        margin-bottom:10px;
                        border-left:5px solid #38bdf8;
                    ">
                        <b>{row['Categoría']}</b><br>
                        Cantidad: {row['Cantidad']}<br>
                        Confianza promedio: {row['Confianza promedio']}
                    </div>
                    """, unsafe_allow_html=True)

                # 📈 GRÁFICA
                st.markdown("### 📈 Distribución")
                st.bar_chart(
                    df.set_index("Categoría")["Cantidad"],
                    use_container_width=True
                )

            else:
                st.warning("⚠️ No se detectaron objetos.")
                st.caption("Prueba bajando el umbral de confianza.")

else:
    st.error("❌ No se pudo cargar el modelo.")
    st.stop()

# 🧾 FOOTER
st.markdown("""
<hr>
<p style='text-align:center; font-size:14px; color:gray;'>
Hecho con ❤️ usando YOLOv5 + Streamlit + PyTorch
</p>
""", unsafe_allow_html=True)
