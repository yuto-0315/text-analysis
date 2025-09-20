# text-analysis

このプロジェクトの簡単な説明

## Python3.11.x インストール方法
https://www.python.org/downloads/windows/


## Python 仮想環境 (venv)
このプロジェクトではリポジトリルートに `.venv` という名前で Python の仮想環境を作成することを想定しています。以下は macOS (zsh) での手順です。

### 仮想環境の作成
```bash
cd /text-analysis
python3 -m venv .venv
```

### 仮想環境の有効化（activate）
仮想環境を有効化すると、そのシェル内でインストールしたパッケージや Python が優先されます。zsh の場合:

```bash
source .venv/bin/activate
# プロンプトが変わり、( .venv ) のように表示されます
python -V
pip -V
```

### コードの実行
```bash
streamlit run text_analyzer_app.py
```

### 仮想環境の終了（deactivate）
仮想環境を終了するには:

```bash
deactivate
```

### 依存関係の保存
仮想環境を有効化した状態でインストールした依存を保存するには:

```bash
pip freeze > requirements.txt
```

### 依存関係の復元
仮想環境を作成・有効化した上で、requirements.txt から依存をインストールします。手順例（macOS / zsh）:

```bash
# pip を最新にしてから依存をインストール
pip install --upgrade pip
pip install -r requirements.txt
```

requirements.txt がない場合は、依存を手動でインストールしてから `pip freeze > requirements.txt` を実行して保存してください。

### 備考
- VS Code をお使いの場合は、ワークスペースで `.venv` を選択することで自動的にその仮想環境を使用できます。
- 他のシェル（bash, fish 等）を使う場合は有効化コマンドが異なる場合があります。
