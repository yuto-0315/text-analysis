import streamlit as st
import pandas as pd
from janome.tokenizer import Tokenizer
import matplotlib.pyplot as plt
import japanize_matplotlib
import networkx as nx
from collections import Counter
import re
import prince
import json

# --- アプリケーションの基本設定 ---
st.set_page_config(
    page_title="テキスト分析ツール",
    layout="wide"
)

# --- 定数定義 ---
PRESET_RULES = {
    # "感情": {"コード名": "＊感情", "単語リスト": "嬉しい,楽しい,悲しい,怒り,好き,嫌い,安心,不安,喜び,驚き,落ち着く,寂しい,興奮"},
    # "人物": {"コード名": "＊人物", "単語リスト": "私,あなた,彼,彼女,友人,家族,先生,自分,相手,人間,同僚,上司,部下,子供,親"},
    # "時間": {"コード名": "＊時間", "単語リスト": "時,時間,今日,明日,昨日,未来,過去,現在,昔,今,朝,昼,夜,週,月"},
    # "行動": {"コード名": "＊行動", "単語リスト": "見る,聞く,話す,行く,来る,食べる,思う,感じる,考える,する,始める,続ける,止める,決める"},
    # "場所": {"コード名": "＊場所", "単語リスト": "家,学校,職場,公園,店,駅,部屋,街,庭,海,山,病院,図書館,カフェ,会場"},
    # "状況": {"コード名": "＊状況", "単語リスト": "問題,事情,状況,機会,危機,成功,失敗,準備,変化,過程,理由,条件,背景,影響,対応"},
    # "目的": {"コード名": "＊目的", "単語リスト": "目的,理由,目標,意図,動機,狙い,ため,ために,ゴール,狙う,達成,意向,希望,狙い目"},
    # "評価": {"コード名": "＊評価", "単語リスト": "良い,悪い,優れている,劣っている,高い,低い,満足,不満,評価,批判,賞賛,問題点,利点,欠点"},
    # "健康": {"コード名": "＊健康", "単語リスト": "健康,病気,体調,疲れ,睡眠,運動,食事,治療,診察,薬,検査,怪我,ストレス,回復"},
    # "仕事": {"コード名": "＊仕事", "単語リスト": "仕事,業務,職場,転職,残業,会議,プロジェクト,給料,雇用,担当,評価,納期,契約,部署"},
    # "数量": {"コード名": "＊数量", "単語リスト": "多い,少ない,数,量,倍,全部,一部,一つ,二つ,半分,程度,増える,減る,平均,割合"},
    "自然": {"コード名": "＊自然", "単語リスト": "光,花,風,星空,雨,海,山,川,河,丘,雪,波,葉,雲,青空,虹,潮,天,森,林,木,欅,太陽,朝日,夕日,大地,地,土,水,地球,宇宙,草,実,泉,月,畑,種,野,火,野原,高原,炎"},
    "人間関係": {"コード名": "＊人間関係", "単語リスト": "もの,者,われ,我,友情,自分,じぶん,彼,あなた,君,きみ,父,母,僕,ぼく,私,わたし,人,人間,友だち,ともだち,友達,友人,仲間,家族,なかま,かぞく,みんな,おじさん,おばさん,誰,だれ"},
    "環境生活": {"コード名": "＊環境生活", "単語リスト": "学校,世界,世,町,まち,街"},
    "時間2": {"コード名": "＊時間2", "単語リスト": "日,時,とき,時間,朝,昼,夕,夜,昨日,今日,明日,未来,過去,昔,春,夏,秋,冬"},
    "心情": {"コード名": "＊心情", "単語リスト": "思い出,想い出,思い出す,こころ,心,気持ち"},
    "ポジティブ": {"コード名": "＊ポジティブ", "単語リスト": "夢,希望,遥か,彼方,愛,志,願い,勇気,絆,奇跡,道,途,路,扉,歌,命,無限,銀河,平和,生命,生きる,輝き,祈り,祈る,力,羽,翼,幸せ,幸福,遙か,歓び,信じる,笑顔,笑う,輝く,青春,光る,よろこび"},
    "ネガティブ": {"コード名": "＊ネガティブ", "単語リスト": "悲しい,哀しい,悲しみ,哀しみ,悲しむ,哀しむ,涙,泣く,別れ,かなしい,かなしみ,かなしむ,なみだ,なく,わかれ,弱さ,よわさ,弱い,よわい,寂しい,さびしい,寂しさ,さびしさ,不安"},
    "体": {"コード名": "＊体", "単語リスト": "頭,あたま,口,くち,顔,かお,耳,みみ,目,瞳,声,こえ,肩,かた,足,あし,脚,胸,むね,頬,腕,ひとみ"},
    "色": {"コード名": "＊色", "単語リスト": "青,青い,赤,赤い,白,白い,黄,黄色,黄色い,黒,黒い,緑,緑色,虹色,金,金色,銀,銀色,色"},
}

# --- Streamlitのセッション状態管理 ---
if 'documents' not in st.session_state:
    st.session_state.documents = [{"title": "自己紹介", "text": "これはなかたにが書いたテキスト分析ツールだよ。\nこんな感じに文章を追加していけるよ。"}]

if 'rules_df' not in st.session_state:
    st.session_state.rules_df = pd.DataFrame(columns=['コード名', '単語リスト'])

# ★ 追加：分析結果を保存するための場所を初期化
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

# --- 関数の定義 ---
@st.cache_resource
def get_tokenizer():
    return Tokenizer()

def add_document():
    doc_count = len(st.session_state.documents)
    st.session_state.documents.append({"title": f"文書{doc_count + 1}", "text": ""})

def remove_document(index):
    st.session_state.documents.pop(index)

def add_preset_rule(name):
    new_rule = pd.DataFrame([PRESET_RULES[name]])
    if 'コード名' not in st.session_state.rules_df.columns or '単語リスト' not in st.session_state.rules_df.columns:
        st.session_state.rules_df = pd.DataFrame(columns=['コード名', '単語リスト'])
    new_code = new_rule['コード名'].iloc[0]
    if not st.session_state.rules_df['コード名'].fillna('').eq(new_code).any():
        st.session_state.rules_df = pd.concat([st.session_state.rules_df, new_rule], ignore_index=True)

def analyze(documents, rules_df):
    tokenizer = get_tokenizer()
    
    morph_results = {}
    for doc in documents:
        title = doc['title'].strip()
        text = doc['text']
        if not title or not text.strip(): continue
        tokens = tokenizer.tokenize(text)
        morph_list = []
        for token in tokens:
            morph_list.append({'表層形': token.surface, '品詞': token.part_of_speech.split(',')[0],
                               '品詞細分類1': token.part_of_speech.split(',')[1], '基本形': token.base_form, '読み': token.reading})
        morph_results[title] = pd.DataFrame(morph_list)

    coding_rules = {}
    for _, row in rules_df.iterrows():
        code_name_val, words_str_val = row.get('コード名'), row.get('単語リスト')
        if pd.notna(code_name_val) and isinstance(code_name_val, str) and code_name_val.strip():
            code_name = code_name_val.strip()
            if pd.notna(words_str_val) and isinstance(words_str_val, str) and words_str_val.strip():
                coding_rules[code_name] = [w.strip() for w in words_str_val.split(',')]

    if not coding_rules:
        return {"morph_results": morph_results}

    doc_results, coded_word_details = {}, {code: [] for code in coding_rules.keys()}
    for title, df_morph in morph_results.items():
        coded_words_in_doc = []
        for _, token_row in df_morph.iterrows():
            if token_row['品詞'] in ['名詞', '動詞', '形容詞']:
                for code_name, rule_words in coding_rules.items():
                    if token_row['基本形'] in rule_words:
                        coded_words_in_doc.append(code_name)
                        coded_word_details[code_name].append(token_row['基本形'])
                        break
        doc_results[title] = coded_words_in_doc

    if not any(doc_results.values()):
        st.warning("コーディングルールに合致する単語がありませんでした。")
        return {"morph_results": morph_results}

    all_coded_words = [word for words in doc_results.values() for word in words]
    simple_counts = Counter(all_coded_words)
    df_simple = pd.DataFrame(simple_counts.items(), columns=['コード', '出現回数']).sort_values('出現回数', ascending=False)
    
    all_codes = list(coding_rules.keys())
    cross_data = [{'文書タイトル': title, **{code: Counter(codes).get(code, 0) for code in all_codes}} for title, codes in doc_results.items()]
    df_cross = pd.DataFrame(cross_data).set_index('文書タイトル')
    
    return {"df_simple": df_simple, "df_cross": df_cross, "doc_coded_words": list(doc_results.values()), 
            "all_codes": all_codes, "morph_results": morph_results, "coded_word_details": coded_word_details}

# --- データのエクスポート / インポート ---
def make_export_payload():
    # documents と rules_df を JSON 化する
    payload = {
        "documents": st.session_state.get('documents', []),
        "rules": st.session_state.get('rules_df', pd.DataFrame(columns=['コード名', '単語リスト'])).to_dict(orient='records')
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)

def load_from_payload(payload_str):
    try:
        data = json.loads(payload_str)
    except Exception as e:
        st.error(f"読み込みに失敗しました（JSON解析エラー）: {e}")
        return False

    # バリデーションと復元
    documents = data.get('documents')
    rules = data.get('rules')
    if not isinstance(documents, list) or not isinstance(rules, list):
        st.error("読み込んだデータの形式が不正です。'documents' と 'rules' を含むJSONを指定してください。")
        return False

    # documents を session_state にセット
    st.session_state.documents = []
    for i, d in enumerate(documents):
        title = d.get('title', f"文書{i+1}") if isinstance(d, dict) else f"文書{i+1}"
        text = d.get('text', '') if isinstance(d, dict) else ''
        st.session_state.documents.append({'title': title, 'text': text})

    # rules を DataFrame に変換してセット
    try:
        df_rules = pd.DataFrame(rules)
        # ensure columns
        if 'コード名' not in df_rules.columns or '単語リスト' not in df_rules.columns:
            # try to detect alternative keys
            df_rules = df_rules.rename(columns={c: 'コード名' if 'コード' in c else c for c in df_rules.columns})
        st.session_state.rules_df = df_rules.reindex(columns=['コード名', '単語リスト'])
    except Exception as e:
        st.error(f"ルールの復元に失敗しました: {e}")
        return False

    # title_{i} と text_{i} を session_state に用意しておく
    for i, doc in enumerate(st.session_state.documents):
        st.session_state[f"title_{i}"] = doc['title']
        st.session_state[f"text_{i}"] = doc['text']

    # 不要な以前のキーを消す（もし文書数が減っていた場合）
    existing_title_keys = [k for k in list(st.session_state.keys()) if isinstance(k, str) and k.startswith('title_')]
    for k in existing_title_keys:
        idx = int(k.split('_')[1]) if '_' in k else None
        if idx is not None and idx >= len(st.session_state.documents):
            del st.session_state[k]
    existing_text_keys = [k for k in list(st.session_state.keys()) if isinstance(k, str) and k.startswith('text_')]
    for k in existing_text_keys:
        idx = int(k.split('_')[1]) if '_' in k else None
        if idx is not None and idx >= len(st.session_state.documents):
            del st.session_state[k]

    st.success("データを読み込み、セッションを復元しました。")
    return True

# --- GUIの描画 ---
st.markdown('<h1>テキスト分析<span style="color:#68BCFF; font-size:0.3em; margin-top:-8px;">©nakatani</span></h1>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.header("1. 分析テキスト入力")
    st.subheader("テキストファイルをアップロード (.txt 複数可)")
    uploaded_txts = st.file_uploader("複数のテキストファイルを選択してください", type=['txt'], accept_multiple_files=True)
    if uploaded_txts:
        added = 0
        for f in uploaded_txts:
            try:
                raw = f.read()
                try:
                    text = raw.decode('utf-8')
                except Exception:
                    text = raw.decode('utf-8', errors='ignore')
                # filename から拡張子を除いたものをタイトルにする
                name = f.name
                if name.lower().endswith('.txt'):
                    title = name[:-4]
                else:
                    title = name
                st.session_state.documents.append({'title': title, 'text': text})
                added += 1
            except Exception as e:
                st.error(f"ファイル読み込みに失敗しました: {f.name} ({e})")
        if added > 0:
            st.success(f"{added} 個のテキストファイルを読み込みました。")
            st.experimental_rerun()
    for i, doc in enumerate(st.session_state.documents):
        with st.container(border=True):
            st.text_input("タイトル", value=doc["title"], key=f"title_{i}")
            st.text_area("本文", value=doc["text"], key=f"text_{i}", height=150)
            st.button("➖ この文書を削除", key=f"remove_{i}", on_click=remove_document, args=(i,))
    st.button("➕ 文書を追加", on_click=add_document)

with col2:
    st.header("2. コーディングルール設定")
    st.subheader("テンプレートから追加")
    num_cols = 6
    rows_of_buttons = [list(PRESET_RULES.keys())[i:i + num_cols] for i in range(0, len(PRESET_RULES), num_cols)]
    for row in rows_of_buttons:
        cols_preset = st.columns(num_cols)
        for i, name in enumerate(row):
            with cols_preset[i]:
                # ★ 修正：use_container_width を削除
                st.button(name, on_click=add_preset_rule, args=(name,))

    st.subheader("ルールを編集")
    edited_df = st.data_editor(st.session_state.rules_df, num_rows="dynamic", use_container_width=True)
    st.session_state.rules_df = edited_df

    st.markdown("---")
    st.subheader("設定の保存 / 読み込み")
    col_save, col_load = st.columns([1, 1])
    with col_save:
        export_payload = make_export_payload()
        st.download_button("設定をダウンロード (JSON)", data=export_payload, file_name="text_analysis_export.json", mime='application/json')
    with col_load:
        uploaded = st.file_uploader("保存したJSONをアップロードして復元", type=['json'])
        if uploaded is not None:
            try:
                payload_str = uploaded.read().decode('utf-8')
            except Exception:
                payload_str = uploaded.read().decode('utf-8', errors='ignore')
            if load_from_payload(payload_str):
                # reload: simply rerun by setting a small placeholder; Streamlit will re-run automatically
                st.experimental_rerun()

# ★ 修正：use_container_width を削除
if st.button("分析開始", type="primary"):
    current_documents = [{"title": st.session_state[f"title_{i}"], "text": st.session_state[f"text_{i}"]} for i in range(len(st.session_state.documents))]
    if not any(doc['text'].strip() for doc in current_documents):
        st.error("分析するテキストを入力してください。")
    else:
        with st.spinner('分析を実行中...'):
            # ★ 修正：結果をセッション状態に保存
            st.session_state.analysis_result = analyze(current_documents, st.session_state.rules_df)

# ★ 修正：セッション状態に結果があれば、結果セクションを描画
if st.session_state.analysis_result:
    st.header("3. 分析結果")
    
    # 分析結果をセッション状態から取り出す
    result = st.session_state.analysis_result
    morph_results = result.get("morph_results")
    
    # コーディングが行われたかどうかで表示するタブを切り替える
    if "df_simple" in result:
        st.success("分析が完了しました。")
        df_simple, df_cross, doc_coded_words, all_codes, coded_word_details = \
            result["df_simple"], result["df_cross"], result["doc_coded_words"], result["all_codes"], result["coded_word_details"]
        
        tabs = st.tabs(["単純集計", "コード詳細", "クロス集計", "共起ネットワーク", "対応分析", "形態素分析結果"])
        
        with tabs[0]:
            st.subheader("単純集計 (コード出現数)")
            st.dataframe(df_simple, use_container_width=True)
        with tabs[1]:
            st.subheader("コードごとの単語内訳")
            selected_code = st.selectbox("内訳を表示するコードを選択", options=all_codes)
            if selected_code:
                word_counts = Counter(coded_word_details[selected_code])
                df_details = pd.DataFrame(word_counts.items(), columns=['単語 (基本形)', '出現回数']).sort_values('出現回数', ascending=False)
                st.dataframe(df_details, use_container_width=True)
        with tabs[2]:
            st.subheader("クロス集計 (文書ごとのコード出現数)")
            st.dataframe(df_cross, use_container_width=True)
            if not df_cross.empty:
                fig, ax = plt.subplots(figsize=(10, df_cross.shape[0] * 0.5 + 3))
                df_cross.plot(kind='barh', stacked=True, ax=ax)
                ax.set_title("文書ごとのコード出現数")
                ax.set_xlabel("出現回数")
                st.pyplot(fig)
        with tabs[3]:
            st.subheader("コードの共起ネットワーク")
            co_occurrence = {c1: {c2: 0 for c2 in all_codes} for c1 in all_codes}
            for codes in doc_coded_words:
                unique_codes = sorted(list(set(codes)))
                for i in range(len(unique_codes)):
                    for j in range(i + 1, len(unique_codes)):
                        c1, c2 = unique_codes[i], unique_codes[j]
                        co_occurrence[c1][c2] += 1
                        co_occurrence[c2][c1] += 1
            G = nx.Graph()
            for c1, others in co_occurrence.items():
                for c2, weight in others.items():
                    if weight > 0: G.add_edge(c1, c2, weight=weight)
            if G.number_of_nodes() > 0:
                fig, ax = plt.subplots(figsize=(10, 10))
                pos = nx.spring_layout(G, k=1, seed=42)
                weights = [G[u][v]['weight'] for u, v in G.edges()]
                edge_width = [w * 1.5 for w in weights]
                nx.draw_networkx(G, pos, ax=ax, with_labels=True, node_size=3500, node_color='skyblue',
                                 font_family='IPAexGothic', font_size=12, width=edge_width, edge_color='gray')
                ax.set_title("コードの共起ネットワーク")
                st.pyplot(fig)
            else: st.warning("共起ネットワークを描画できるコードの組み合わせがありませんでした。")
        with tabs[4]:
            st.subheader("対応分析 (文書とコードの関係性の可視化)")
            df_cross_filtered = df_cross.loc[df_cross.sum(axis=1) > 0, df_cross.sum(axis=0) > 0]
            if df_cross_filtered.shape[0] < df_cross.shape[0] or df_cross_filtered.shape[1] < df_cross.shape[1]:
                st.warning("出現数が0の文書またはコードがあったため、対応分析の対象から除外しました。")
            if df_cross_filtered.shape[0] >= 2 and df_cross_filtered.shape[1] >= 2:
                try:
                    ca = prince.CA(n_components=2, n_iter=3, copy=True, check_input=True, engine='sklearn', random_state=42).fit(df_cross_filtered)
                    altair_chart = ca.plot(df_cross_filtered).properties(title="対応分析プロット", width=600, height=500)
                    st.altair_chart(altair_chart, use_container_width=True)
                    st.info("""
                        **【プロットの見方】**
                        - **近くにある点同士は、関連性が強い**ことを示します。
                        - **点（円）**は行（文書タイトル）、**点（三角）**は列（コード）を表します。
                        - 例えば、ある文書（円）とあるコード（三角）が近くにあれば、その文書ではそのコードが特徴的に多く出現していると解釈できます。
                        - 原点（0, 0）から離れている点ほど、全体平均から見て特徴的な傾向を持つことを示唆します。
                    """)
                except Exception as e: st.error(f"対応分析の実行中にエラーが発生しました: {e}")
            else: st.warning("対応分析を実行するには、出現数が0でない文書とコードがそれぞれ2つ以上必要です。")
        with tabs[5]:
            st.subheader("形態素分析結果")
            st.info("各文書がどのように単語に分解されたかを確認できます。")
            doc_titles = list(morph_results.keys())
            if doc_titles:
                selected_title = st.selectbox("結果を表示する文書を選択してください", options=doc_titles, key="morph_select")
                if selected_title: st.dataframe(morph_results[selected_title], use_container_width=True)
    else: # コーディングルールがない場合
        st.success("形態素分析が完了しました。")
        tab1, = st.tabs(["形態素分析結果"])
        with tab1:
            st.subheader("形態素分析結果")
            st.info("各文書がどのように単語に分解されたかを確認できます。")
            doc_titles = list(morph_results.keys())
            if doc_titles:
                selected_title = st.selectbox("結果を表示する文書を選択してください", options=doc_titles, key="morph_select_only")
                if selected_title: st.dataframe(morph_results[selected_title], use_container_width=True)