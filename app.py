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

# 🎨 ESTILO SEGURO (NO ROMPE STREAMLIT)
st.markdown("""
<style>

/* Fondo general */
.stApp {
    background-color: #0f172a;
}

/* Contenedor principal tipo tarjeta */
.block-container {
    padding: 2rem;
    border-radius: 12px;
}

/* Títulos */
h1 {
    text-align: center;
    color: #38bdf8;
}

h2, h3 {
    color: #e2e8f0;
}

/* Texto */
p, label {
    color: #cbd5f5;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #020617;
}

/* Tarjetas */
.card {
    background-color: #1e293b;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 10px;
    border-left: 4px solid #38bdf8;
}

</style>
""", unsafe_allow_html=True)

# 🤖 MODELO
@st.cache_resource
def load_model():
    try:
        from ultralytics import YOLO
        return YOLO("yolov5su.pt")
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        return None

# 🧠 HEADER
st.markdown("<h1>🔍 ¿Qué estoy viendo?</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>Detecta objetos en tiempo real usando inteligencia artificial</p>",
    unsafe_allow_html=True
)

with st.spinner("Cargando modelo..."):
    model = load_model()

# ⚙️ SIDEBAR
if model:
    with st.sidebar:
        st.markdown("## ⚙️ Configuración")
        st.markdown("---")

        conf_threshold = st.slider("Confianza mínima", 0.0, 1.0, 0.25, 0.01)
        iou_threshold  = st.slider("Umbral IoU", 0.0, 1.0, 0.45, 0.01)
        max_det        = st.number_input("Máx detecciones", 10, 2000, 1000, 10)

        st.markdown("---")
        st.caption("Ajusta para mejorar resultados")

    # 📸 INPUTS (CÁMARA + UPLOAD)
    st.markdown("### 📸 Captura o sube una imagen")

    col_input1, col_input2 = st.columns(2)

    with col_input1:
        picture = st.camera_input("Tomar foto")

    with col_input2:
        uploaded = st.file_uploader("Subir imagen", type=["jpg", "png", "jpeg"])

    if picture:
        bytes_data = picture.getvalue()
    elif uploaded:
        bytes_data = uploaded.read()
    else:
        bytes_data = None

    if bytes_data:
        pil_img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
        np_img  = np.array(pil_img)[..., ::-1]

        with st.spinner("🧠 Analizando imagen..."):
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

        result = results[0]
        boxes = result.boxes
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
                label_names = model.names
                category_count = {}
                category_conf = {}

                for box in boxes:
                    cat = int(box.cls.item())
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

                # 🎯 TARJETAS
                for row in data:
                    st.markdown(f"""
                    <div class="card">
                        <b>{row['Categoría']}</b><br>
                        Cantidad: {row['Cantidad']}<br>
                        Confianza: {row['Confianza promedio']}
                    </div>
                    """, unsafe_allow_html=True)

                # 📈 GRÁFICA
                st.markdown("### 📈 Distribución")
                st.bar_chart(df.set_index("Categoría")["Cantidad"])

            else:
                st.warning("No se detectaron objetos")
                st.caption("Prueba bajar la confianza")

else:
    st.error("No se pudo cargar el modelo")
    st.stop()

# 🧾 FOOTER
st.markdown("""
<hr>
<p style='text-align:center; color:gray;'>
Hecho con ❤️ usando YOLO + Streamlit
</p>
""", unsafe_allow_html=True)
