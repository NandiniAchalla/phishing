def read_html(html_path):
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()
