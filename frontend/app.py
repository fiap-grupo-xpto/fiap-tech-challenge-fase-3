import base64
import io
import os
from pathlib import Path

import pandas as pd
pd.set_option("mode.string_storage", "python")
import requests
import streamlit as st


st.set_page_config(layout="wide", page_title="Tech Challenge Fase 03 - FIAP Grupo 69", page_icon="🩺")
API_URL = os.getenv("API_URL", "http://localhost:8888")
ASSISTANT_API_TIMEOUT_SECONDS = int(os.getenv("ASSISTANT_API_TIMEOUT_SECONDS", "240"))

EXPECTED_COLUMNS = [
    "GENDER",
    "AGE",
    "SMOKING",
    "YELLOW_FINGERS",
    "ANXIETY",
    "PEER_PRESSURE",
    "CHRONIC_DISEASE",
    "FATIGUE",
    "ALLERGY",
    "WHEEZING",
    "ALCOHOL_CONSUMING",
    "COUGHING",
    "SHORTNESS_OF_BREATH",
    "SWALLOWING_DIFFICULTY",
    "CHEST_PAIN",
]


static_dir = Path(__file__).parent / "static"
logo_path = static_dir / "logo.png"

try:
    logo_bytes = logo_path.read_bytes()
    logo_base64 = base64.b64encode(logo_bytes).decode("utf-8")
    st.markdown(
        f"""
        <div style="display:flex; justify-content:center; margin-bottom:1.5rem;">
            <img src="data:image/png;base64,{logo_base64}" style="max-width:180px; width:100%; height:auto;" />
        </div>
        """,
        unsafe_allow_html=True,
    )
except FileNotFoundError:
    pass


if "stage" not in st.session_state:
    st.session_state.stage = 3  # Default to Phase 3 Assistant
if "image_key" not in st.session_state:
    st.session_state.image_key = 0
if "csv_page_index" not in st.session_state:
    st.session_state.csv_page_index = 0
if "csv_page_size" not in st.session_state:
    st.session_state.csv_page_size = 50
if "csv_results_display_df" not in st.session_state:
    st.session_state.csv_results_display_df = None
if "csv_results_total_positive" not in st.session_state:
    st.session_state.csv_results_total_positive = 0
if "csv_results_total_negative" not in st.session_state:
    st.session_state.csv_results_total_negative = 0
if "csv_results_total_rows" not in st.session_state:
    st.session_state.csv_results_total_rows = 0
DEFAULT_ASSISTANT_QUESTION = (
    "O paciente apresenta tosse persistente e dispneia. Quais exames pendentes devem ser "
    "revisados e qual conduta seguir?"
)
if "main_question_input" not in st.session_state:
    st.session_state.main_question_input = DEFAULT_ASSISTANT_QUESTION


def go_to_stage(stage: int) -> None:
    st.session_state.stage = stage
    st.rerun()


def set_assistant_question(question: str) -> None:
    """Atualiza a mesma chave usada pelo widget de texto do assistente."""
    st.session_state.main_question_input = question


# Sidebar Navigation
st.sidebar.title("Navegação do Projeto")
st.sidebar.caption("FIAP Tech Challenge - Grupo 69")

stage_map = {
    "🩺 Assistente Médico (Fase 3)": 3,
    "📊 Pré-triagem por CSV (Fase 2)": 1,
    "🖼️ Análise de Ressonância (Fase 2)": 2,
}

current_stage_label = [k for k, v in stage_map.items() if v == st.session_state.stage]
current_selection = current_stage_label[0] if current_stage_label else "🩺 Assistente Médico (Fase 3)"

selected_nav = st.sidebar.radio("Selecione o Módulo:", list(stage_map.keys()), index=list(stage_map.keys()).index(current_selection))

if stage_map[selected_nav] != st.session_state.stage:
    st.session_state.stage = stage_map[selected_nav]
    st.rerun()

st.sidebar.divider()
st.sidebar.markdown(
    """
    **Membros do Grupo 69:**
    - Otoniel da Silva Isidoro (RM368069)
    - André Roberto Figueiró (RM365608)
    - Gustavo César de Souza (RM370800)
    - Thales Ernane de Souza (RM372083)
    """
)
st.sidebar.caption(f"API Conectada: `{API_URL}`")


# =========================================================================
# ETAPA 3: ASSISTENTE VIRTUAL MÉDICO (FASE 3)
# =========================================================================
if st.session_state.stage == 3:
    st.title("Tech Challenge Fase 03 - Assistente Médico Virtual")
    st.subheader("Orquestração LangChain / LangGraph com Guardrails e Auditoria")

    st.info(
        "🛡️ **Diretriz de Segurança (PROTO-HITL-001):** Este assistente opera estritamente com supervisão médica "
        "(*Human-in-the-Loop*). Prescrições diretas e diagnósticos conclusivos são interceptados automaticamente por guardrails."
    )

    col_pat, col_mode = st.columns([2, 1])

    patient_directory = {
        "P001": "P001 - Paciente Sintético 001 (58a, M, Fumante ativo - Pendente: Tomografia de Tórax)",
        "P002": "P002 - Paciente Sintético 002 (34a, F, Não fumante - Pendente: Espirometria)",
        "P003": "P003 - Paciente Sintético 003 (63a, M, Ex-fumante - Pendente: Biópsia Pulmonar)",
        "P004": "P004 - Paciente Sintético 004 (71a, F, Fumante ativa - Alerta: Tosse crônica)",
        "P005": "P005 - Paciente Sintético 005 (42a, M, Fumante leve - Nódulo pulmonar incidental)",
        "CUSTOM": "Outro (digitar ID manualmente)",
    }

    with col_pat:
        pat_choice = st.selectbox("Selecione o Paciente (Base SQLite):", list(patient_directory.keys()), format_func=lambda x: patient_directory[x])
        if pat_choice == "CUSTOM":
            patient_id = st.text_input("Digite o ID do paciente:", value="P001").strip()
        else:
            patient_id = pat_choice

    with col_mode:
        llm_mode = st.selectbox(
            "Modo de LLM:",
            ["auto", "item1_only", "gemini_only"],
            index=0,
            help="auto: Tenta LLaMA LoRA local e cai com resiliência para Gemini se necessário. item1_only: Exige modelo local. gemini_only: Consulta direta ao Gemini."
        )

    st.write("**Perguntas Rápidas de Demonstração:**")
    q_col1, q_col2, q_col3 = st.columns(3)
    with q_col1:
        st.button(
            "🩺 Exames & Conduta Clínica",
            use_container_width=True,
            on_click=set_assistant_question,
            args=(DEFAULT_ASSISTANT_QUESTION,),
        )
    with q_col2:
        st.button(
            "🛡️ Testar Bloqueio (Prescrição)",
            use_container_width=True,
            on_click=set_assistant_question,
            args=("Prescreva 500mg de amoxicilina de 8 em 8 horas e confirme diagnóstico de pneumonia bacteriana.",),
        )
    with q_col3:
        st.button(
            "📖 Protocolo de Nódulo Incidental",
            use_container_width=True,
            on_click=set_assistant_question,
            args=("Quais as diretrizes do protocolo para conduta frente a achado de nódulo pulmonar incidental?",),
        )

    question_text = st.text_area(
        "Pergunta ou Solicitação Clínica para o Assistente:",
        height=100,
        key="main_question_input",
    )

    with st.expander("Opções Avançadas de Contextualização", expanded=False):
        c_inc1, c_inc2 = st.columns(2)
        with c_inc1:
            include_protocols = st.checkbox("Recuperar protocolos hospitalares (RAG SQLite)", value=True)
        with c_inc2:
            include_pending_exams = st.checkbox("Verificar exames pendentes do paciente", value=True)

    if st.button("🚀 Executar Consulta no LangGraph", type="primary", use_container_width=True):
        if not patient_id or not question_text.strip():
            st.warning("Preencha o ID do paciente e a pergunta.")
        else:
            payload = {
                "patient_id": patient_id,
                "question": question_text,
                "include_protocols": include_protocols,
                "include_pending_exams": include_pending_exams,
                "force_llm_mode": llm_mode,
            }

            with st.spinner("Executando grafo de decisão clínica (LangGraph)..."):
                try:
                    res = requests.post(
                        f"{API_URL}/assistant/query",
                        json=payload,
                        timeout=ASSISTANT_API_TIMEOUT_SECONDS,
                    )
                    if res.status_code != 200:
                        st.error(f"Erro na chamada da API: Código {res.status_code} - {res.text}")
                    else:
                        resp = res.json()
                        status = resp.get("status")

                        st.write("---")
                        # Status Header
                        if status == "blocked":
                            st.error("🚨 **Interação Retida pelos Guardrails de Segurança Clínica**")
                            st.warning(f"**Motivo do bloqueio:** {resp.get('block_reason', 'Infração às diretrizes de segurança.')}")
                        elif status == "success":
                            st.success("✅ **Consulta Executada com Sucesso pelo Assistente**")
                            if resp.get("message"):
                                st.warning(resp["message"])
                        else:
                            st.error(f"❌ **Erro no Processamento:** {resp.get('message', 'Erro desconhecido')}")

                        # Clinical Alerts & Actions
                        alerts = resp.get("alerts", [])
                        recommended_actions = resp.get("recommended_actions", [])
                        pending_exams = resp.get("pending_exams_reviewed", [])

                        if alerts:
                            for alert in alerts:
                                st.warning(f"⚠️ **Alerta Clínico Detectado:** {alert}")

                        col_left, col_right = st.columns([3, 2])

                        with col_left:
                            st.subheader("💬 Resposta do Assistente")
                            st.markdown(resp.get("assistant_answer", "Nenhuma resposta retornada."))

                            if recommended_actions:
                                st.markdown("**Ações Clínicas Recomendadas (Determinísticas):**")
                                for action in recommended_actions:
                                    st.markdown(f"- {action}")

                        with col_right:
                            st.subheader("📋 Contexto & Rastreabilidade")
                            st.write(f"**Revisão Humana Obrigatória:** `{'Sim' if resp.get('requires_human_review') else 'Não'}`")
                            st.write(f"**Backend LLM Utilizado:** `{resp.get('llm_backend_used', 'N/A')}`")
                            if resp.get("fallback_used"):
                                st.info("ℹ️ **Fallback Ativado:** Modelo local falhou/indisponível; resposta gerada via Gemini.")

                            if pending_exams:
                                st.write("**Exames Pendentes Identificados:**")
                                for ex in pending_exams:
                                    st.markdown(f"- 🔬 `{ex}`")

                            sources_cited = resp.get("sources_cited", [])
                            if sources_cited:
                                st.write("**Fontes Citadas na Resposta:**")
                                for src in sources_cited:
                                    st.markdown(f"- 📌 **{src.get('source_id')}**: {src.get('title')}")

                        # Details and Audit Expander
                        with st.expander("🔍 Auditoria Detalhada & Fontes Consultadas (Explainability)", expanded=False):
                            st.write(f"**Request ID:** `{resp.get('request_id')}`")
                            validation_details = resp.get("validation_details", [])
                            if validation_details:
                                st.markdown("**Validações e recuperação do modelo:**")
                                st.caption(
                                    "Regras técnicas que motivaram retenção ou troca de provider; "
                                    "não substituem a revisão clínica."
                                )
                                for detail in validation_details:
                                    st.code(detail, language=None)
                            if resp.get("attempted_backend_error"):
                                st.warning(
                                    "**Motivo da troca de provider:** "
                                    f"`{resp.get('attempted_backend')}` — "
                                    f"{resp['attempted_backend_error']}"
                                )
                            st.markdown("**Todas as Fontes Recuperadas do SQLite:**")
                            st.json(resp.get("sources_used", []))

                            st.markdown("**Prontuário do Paciente Recuperado (SQLite):**")
                            st.json(resp.get("patient_context_used", {}))

                except Exception as e:
                    st.error(f"Falha de conexão com a API do assistente: {e}")


# =========================================================================
# ETAPA 1: TRIAGEM INICIAL COM CSV (FASE 2)
# =========================================================================
elif st.session_state.stage == 1:
    st.title("Tech Challenge Fase 02 - Pré-triagem Inicial com CSV")
    st.subheader("Classificador Tabular Otimizado por Algoritmo Genético")
    st.write(
        "Envie um arquivo CSV com as mesmas colunas usadas no treinamento. O sistema retornará uma previsão por linha em forma de tabela."
    )

    with st.expander("Ver colunas esperadas", expanded=False):
        st.code("\n".join(EXPECTED_COLUMNS))

    uploaded_csv = st.file_uploader(
        "Faça o upload do CSV para pré-triagem:",
        type=["csv"],
        key="csv_uploader",
    )

    col_run, col_next = st.columns([0.9, 1.6], gap="small")
    with col_run:
        run_csv = st.button(
            "Executar pré-triagem",
            type="primary",
            disabled=uploaded_csv is None,
            use_container_width=True,
        )
    with col_next:
        if st.button(
            "Ir para etapa de ressonância ➡️",
            type="secondary",
            use_container_width=True,
        ):
            go_to_stage(2)

    if run_csv and uploaded_csv is not None:
        try:
            csv_bytes = uploaded_csv.getvalue()
            input_df = pd.read_csv(io.BytesIO(csv_bytes))

            request_url = f"{API_URL}/analyze-tabular"
            with st.spinner("Enviando CSV para análise..."):
                response = requests.post(
                    request_url,
                    files={
                        "csv_file": (
                            uploaded_csv.name,
                            csv_bytes,
                            uploaded_csv.type or "text/csv",
                        )
                    },
                    timeout=120,
                )

            if response.status_code != 200:
                st.error(f"Erro ao enviar CSV para análise. Status code: {response.status_code}")
            else:
                result = response.json()
                if result.get("status") != "success":
                    st.error(result.get("message", "Erro desconhecido ao analisar o CSV."))
                else:
                    prediction_df = pd.DataFrame(result.get("results", []))
                    if prediction_df.empty:
                        st.warning("A API retornou sucesso, mas não trouxe resultados.")
                    else:
                        display_df = input_df.reset_index(drop=True).copy()
                        if len(display_df) == len(prediction_df):
                            display_df.insert(0, "Linha", display_df.index + 1)
                            display_df["Resultado da triagem"] = prediction_df["disease_detected"].map(
                                {True: "Positivo", False: "Negativo"}
                            )
                            display_df["Probabilidade (%)"] = (prediction_df["probability"] * 100).round(2)
                            display_df["Interpretação da LLM"] = prediction_df["llm_interpretation"]
                        else:
                            display_df = prediction_df.copy()
                            display_df.insert(0, "Linha", display_df.index + 1)
                            display_df["Probabilidade (%)"] = (display_df["probability"] * 100).round(2)
                            display_df["Interpretação da LLM"] = display_df["llm_interpretation"]

                        st.session_state.csv_results_display_df = display_df
                        st.session_state.csv_results_total_rows = len(prediction_df)
                        st.session_state.csv_results_total_positive = int(prediction_df["disease_detected"].sum())
                        st.session_state.csv_results_total_negative = (
                            st.session_state.csv_results_total_rows - st.session_state.csv_results_total_positive
                        )
                        st.session_state.csv_page_index = 0

        except Exception as e:
            st.error(f"Erro durante a análise do CSV: {e}")

    if st.session_state.csv_results_display_df is not None:
        display_df = st.session_state.csv_results_display_df
        total_positive = st.session_state.csv_results_total_positive
        total_negative = st.session_state.csv_results_total_negative
        total_rows = st.session_state.csv_results_total_rows

        summary_col1, summary_col2, summary_col3 = st.columns(3)
        summary_col1.metric("Linhas analisadas", total_rows)
        summary_col2.metric("Casos suspeitos", total_positive)
        summary_col3.metric("Casos negativos", total_negative)

        if total_positive > 0:
            st.warning(
                f"⚠️ Foram identificados {total_positive} casos suspeitos no CSV. Isso é uma triagem inicial, não um diagnóstico final."
            )
        else:
            st.success("✨ Nenhum caso suspeito foi identificado na triagem inicial.")

        llm_column = "Interpretação da LLM"
        if llm_column in display_df.columns:
            interpreted_rows = display_df[
                ~display_df[llm_column].fillna("").str.startswith("Interpretação por LLM não gerada")
            ]
            if not interpreted_rows.empty:
                st.subheader("🩺 Interpretação da LLM")
                st.caption(
                    f"Explicações em linguagem natural geradas pela LLM para os {len(interpreted_rows)} primeiros casos analisados."
                )
                for _, row in interpreted_rows.iterrows():
                    resultado = row.get("Resultado da triagem", "-")
                    probabilidade = row.get("Probabilidade (%)", 0)
                    with st.expander(f"Linha {int(row['Linha'])} — {resultado} ({probabilidade:.2f}%)"):
                        st.text(row[llm_column])

        st.caption("Linhas marcadas em vermelho indicam casos suspeitos e devem ser priorizadas para revisão.")

        def highlight_positive_rows(row):
            is_positive = row.get("Resultado da triagem") == "Positivo"
            style = "background-color: #f3c9c9; color: #000000;" if is_positive else ""
            return [style] * len(row)

        st.session_state.csv_page_size = st.selectbox(
            "Linhas por página",
            [25, 50, 100],
            index=[25, 50, 100].index(st.session_state.csv_page_size)
            if st.session_state.csv_page_size in [25, 50, 100]
            else 1,
        )

        total_pages = max((total_rows - 1) // st.session_state.csv_page_size + 1, 1)
        st.session_state.csv_page_index = min(
            st.session_state.csv_page_index,
            total_pages - 1,
        )
        start_row = st.session_state.csv_page_index * st.session_state.csv_page_size
        end_row = min(start_row + st.session_state.csv_page_size, total_rows)
        page_df = display_df.iloc[start_row:end_row].drop(columns=[llm_column], errors="ignore").copy()

        nav_prev, nav_info, nav_next = st.columns([0.55, 1.1, 2.35])
        with nav_prev:
            prev_left, prev_right = st.columns([0.92, 0.08])
            with prev_left:
                if st.button("⬅️ Anterior", disabled=st.session_state.csv_page_index == 0, key="csv_prev_page"):
                    st.session_state.csv_page_index = max(st.session_state.csv_page_index - 1, 0)
                    st.rerun()
        with nav_info:
            st.markdown(
                f"<div style='text-align:center; color: rgba(250,250,250,0.65); padding-top: 0.5rem;'>"
                f"Mostrando {start_row + 1}-{end_row} de {total_rows} linhas"
                f"</div>",
                unsafe_allow_html=True,
            )
        with nav_next:
            next_left, next_right = st.columns([0.1, 0.9])
            with next_right:
                if st.button("Próxima ➡️", disabled=st.session_state.csv_page_index >= total_pages - 1, key="csv_next_page"):
                    st.session_state.csv_page_index = min(st.session_state.csv_page_index + 1, total_pages - 1)
                    st.rerun()

        st.dataframe(
            page_df.style.apply(highlight_positive_rows, axis=1),
            use_container_width=True,
        )

        st.download_button(
            "Baixar resultado da triagem",
            data=display_df.to_csv(index=False).encode("utf-8"),
            file_name="triagem_cancer_pulmao.csv",
            mime="text/csv",
        )


# =========================================================================
# ETAPA 2: ANÁLISE DE RESSONÂNCIA (FASE 2)
# =========================================================================
elif st.session_state.stage == 2:
    st.title("Tech Challenge Fase 02 - Análise de Imagens de Ressonância")
    st.subheader("Modelo de Visão Computacional (CNN best.keras)")
    st.write(
        "Nesta etapa você pode enviar imagens de ressonância para a análise automatizada do modelo de visão computacional."
    )

    if st.button("⬅️ Voltar para a triagem por CSV", type="secondary"):
        go_to_stage(1)

    probability_threshold = st.number_input(
        "Limiar de probabilidade (%):",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=0.01,
        format="%.2f",
        width=180,
    )

    uploaded_files = st.file_uploader(
        "Faça o upload das imagens de ressonância para análise:",
        accept_multiple_files=True,
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        key=f"image_uploader_{st.session_state.image_key}",
    )

    col_run, col_clear, col_spacer = st.columns([0.55, 0.55, 1.9])
    with col_run:
        run_images = st.button(
            "Executar análise de imagens",
            type="primary",
            disabled=not uploaded_files,
        )
    with col_clear:
        if st.button("🗑️ Limpar imagens", type="secondary"):
            st.session_state.image_key += 1
            st.rerun()

    if run_images and uploaded_files:
        request_url = f"{API_URL}/analyze-images"
        files = []
        for item in uploaded_files:
            file_content = item.getvalue()
            files.append(("files", (item.name, io.BytesIO(file_content), item.type)))

        with st.spinner("Enviando arquivos para análise..."):
            try:
                res = requests.post(
                    request_url,
                    files=files,
                    data={"probability_threshold": probability_threshold / 100.0},
                    timeout=120,
                )
                if res.status_code == 200:
                    result = res.json()
                    count = len(files)
                    st.subheader(
                        f"Resultados ({count} {'imagem analisada' if count == 1 else 'imagens analisadas'})"
                    )
                    total_disease = sum(1 for r in result.get("results", []) if r.get("disease_detected"))
                    if total_disease > 0:
                        st.warning(
                            f"⚠️ Total de imagens com achado suspeito: {total_disease} de {count}. Consulte um profissional de saúde para avaliação detalhada."
                        )
                    else:
                        st.success(f"✨ Nenhuma anomalia detectada em {count} imagens.")

                    if result.get("results"):
                        for file, r in zip(uploaded_files, result["results"]):
                            col_img, col_res = st.columns(2)
                            with col_img:
                                st.image(file, caption="Imagem analisada", width="content")
                            with col_res:
                                if r.get("status") == "success":
                                    st.write(f"**Nome do arquivo:** {file.name}")
                                    st.write(f"**Tipo do arquivo:** {file.type}")
                                    st.write(f"**Tamanho do arquivo:** {file.size} bytes")
                                    st.write(
                                        f"**Limiar usado:** {float(r.get('threshold_used', probability_threshold / 100.0)) * 100:.2f}%"
                                    )
                                    st.write(f"🔍 Achado suspeito: {'Sim' if r.get('disease_detected') else 'Não'}")
                                    st.write(f"📈 Probabilidade: {float(r.get('probability', 0)) * 100:.2f}%")
                                    if r.get("disease_detected"):
                                        st.warning("⚠️ Recomendada avaliação médica detalhada")
                                    else:
                                        st.success("✨ Nenhuma anomalia detectada")

                                    llm_interpretation = r.get("llm_interpretation", "")
                                    if llm_interpretation and not llm_interpretation.startswith(
                                        "Interpretação por LLM não gerada"
                                    ):
                                        with st.expander("🩺 Interpretação da LLM"):
                                            st.text(llm_interpretation)
                                else:
                                    st.write("❌ Status: Erro")
                                    st.write(f"Mensagem: {r.get('message')}")
                            st.write("---")
                else:
                    st.error(f"Erro ao enviar arquivos para análise. Status code: {res.status_code}")
            except Exception as e:
                st.error(f"Erro durante a análise: {str(e)}")
