import csv
import pandas as pd
import streamlit as st

# Amplify関連
from amplify import BinarySymbolGenerator, sum_poly, Solver
# Amplify AE用クライアント
from amplify.client import FixstarsClient

def load_employees_from_csv(csv_file_path):
    """
    CSVファイルから社員データを読み込む
    CSVの想定カラム: 社員番号, リーダー, スキル
    例:
      社員番号,リーダー,スキル
      101,1,7
      102,0,3
      ...
    戻り値: [(社員番号, リーダーフラグ, スキル), ...] のリスト
    """
    employees = []
    with open(csv_file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            emp_id = int(row["社員番号"])
            leader = int(row["リーダー"])
            skill = int(row["スキル"])
            employees.append((emp_id, leader, skill))
    return employees


def solve_skill_partition_multi_group(employees, M, token, pen_group_size=1.0):
    """
    employees: [(社員番号, リーダー(0/1), スキル), ...]
    M: グループ数
    token: Amplify AE のトークン文字列
    pen_group_size: グループ人数差をどの程度ペナルティ化するかの重み

    戻り値:
      (groups, group_skills, group_sizes)
       groups: グループごとの社員番号リスト
       group_skills: グループごとのスキル合計
       group_sizes: グループごとの人数
    """
    N = len(employees)
    if N == 0:
        st.warning("社員データがありません。")
        return [], [], []

    # リーダーの社員インデックス
    leader_indices = [i for i, (_, leader, _) in enumerate(employees) if leader == 1]
    # 各社員のスキル
    skills = [emp[2] for emp in employees]

    # リーダー数 < グループ数 の場合は不可能
    if len(leader_indices) < M:
        st.warning("リーダー数がグループ数より少ないため、各グループにリーダーを配置できません。")
        return [], [], []

    # 1) 変数定義: x[i][k] = 社員iがグループkに所属 (0 or 1)
    gen = BinarySymbolGenerator()
    x = gen.array(N, M)

    # 2) グループごとのスキル合計
    #    S[k] = Σ_i ( skill_i * x[i][k] )
    S = []
    for k in range(M):
        s_k = sum_poly(skills[i] * x[i][k] for i in range(N))
        S.append(s_k)

    # スキル差のコスト: Σ_{k<l}(S[k] - S[l])^2
    skill_diff_cost = 0
    for k in range(M):
        for l in range(k + 1, M):
            diff = S[k] - S[l]
            skill_diff_cost += diff * diff

    # 3) グループごとの人数
    #    N_k = Σ_i x[i][k]
    N_k_list = []
    for k in range(M):
        n_k = sum_poly(x[i][k] for i in range(N))
        N_k_list.append(n_k)

    # 人数差のコスト: Σ_{k<l}(N_k - N_l)^2
    group_size_cost = 0
    for k in range(M):
        for l in range(k+1, M):
            diff = N_k_list[k] - N_k_list[l]
            group_size_cost += diff * diff

    # group_size_cost に重みをかける
    group_size_cost *= pen_group_size

    # 4) 制約ペナルティ
    # (a) 各社員はちょうど1グループ所属 → Σ_k x[i][k] = 1
    PEN_EMP_ONEGROUP = 1000
    employee_penalty = 0
    for i in range(N):
        sum_groups = sum_poly(x[i][k] for k in range(M))
        diff = sum_groups - 1
        employee_penalty += diff * diff
    employee_penalty *= PEN_EMP_ONEGROUP

    # (b) 各グループにリーダー1人以上 → リーダー不在ペナルティ
    PEN_GROUP_LEADER = 1000
    no_leader_pen = 0
    for k in range(M):
        # リーダー不在 → ∏_{i ∈ leader}(1 - x[i][k]) = 1
        prod_term = 1
        for i in leader_indices:
            prod_term *= (1 - x[i][k])
        no_leader_pen += prod_term
    no_leader_pen *= PEN_GROUP_LEADER

    # 5) 総コスト
    cost = skill_diff_cost + group_size_cost + employee_penalty + no_leader_pen

    # 6) Amplify AE クライアント設定
    client = FixstarsClient()
    client.token = token  # 取得したトークンを設定
    client.parameters.timeout = 5 * 1000  # ミリ秒
    solver = Solver(client)

    # 7) ソルバー実行
    result = solver.solve(cost)
    if len(result) == 0:
        st.warning("解が見つかりませんでした(制約が厳しい可能性あり)")
        return [], [], []

    best_values = result[0].values

    # 8) 解釈: 各社員 i について x[i][k]が1のkを所属グループとみなす
    groups = [[] for _ in range(M)]
    group_skills = [0]*M
    group_sizes = [0]*M
    for i, (emp_id, leader, skill) in enumerate(employees):
        assigned_group = None
        for k in range(M):
            if best_values[x[i][k]] == 1:
                assigned_group = k
                break
        if assigned_group is None:
            assigned_group = 0  # 万一見つからない場合

        groups[assigned_group].append(emp_id)
        group_skills[assigned_group] += skill
        group_sizes[assigned_group] += 1

    return groups, group_skills, group_sizes


def main():
    st.title("スキル均等化を目的としたグループ分け")
    token = st.secrets["Your_Amplify_AE_token"]

    # CSVアップロード
    uploaded_file = st.file_uploader("CSVをアップロード", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding="shift-jis")
        st.write("読み込んだデータ:")
        st.dataframe(df.head())

        # グループ数指定
        M = st.number_input("いくつのグループに分けますか？", min_value=2, max_value=20, value=2, step=1)

        # 実行ボタン
        if st.button("最適化を実行"):
            # CSVから (社員番号,リーダー,スキル) のタプルリストへ
            employees_data = []
            for idx, row in df.iterrows():
                emp_id = int(row["社員番号"])
                leader = int(row["リーダー"])
                skill = int(row["スキル"])
                employees_data.append((emp_id, leader, skill))

            # 解を求める
            groups, group_skills, group_sizes = solve_skill_partition_multi_group(
                employees_data, M, token, pen_group_size=1.0
            )

            if groups:
                st.subheader("分割結果")
                for k in range(M):
                    st.write(f"**グループ {k}**")
                    st.write(f"- 社員番号: {groups[k]}")
                    st.write(f"- 人数: {group_sizes[k]}")
                    st.write(f"- スキル合計: {group_skills[k]}")

                if group_skills:
                    max_skill = max(group_skills)
                    min_skill = min(group_skills)
                    diff_skill = max_skill - min_skill
                    st.write(f"スキル合計 → 最大: {max_skill}, 最小: {min_skill}, 差: {diff_skill}")

                if group_sizes:
                    max_size = max(group_sizes)
                    min_size = min(group_sizes)
                    diff_size = max_size - min_size
                    st.write(f"人数 → 最大: {max_size}, 最小: {min_size}, 差: {diff_size}")

    else:
        st.info("上記からCSVファイルをアップロードしてください。")

if __name__ == "__main__":
    main()
