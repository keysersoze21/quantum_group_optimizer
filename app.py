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


def solve_skill_partition_multi_group(employees, M, token):
    """
    与えられた社員リスト(employees)を M グループに分割し、
    Amplify AE を用いて以下を同時に満たす解を探索:
      1) 各社員はちょうど1つのグループに所属
      2) 各グループに最低1人のリーダー(リーダー=1)が含まれる
      3) グループ間のスキル合計差が最小

    employees: [(社員番号,リーダー,スキル), ...]
    M: グループ数 (int)
    token: Amplify AE トークン文字列

    戻り値: (groups, group_skills)
      groups: グループごとに含まれる社員IDリスト (例: [[101,104],[102,103],...])
      group_skills: 各グループのスキル合計 (list of int)
    """
    N = len(employees)
    if N == 0:
        st.warning("社員データが空です。")
        return [], []

    # リーダーのインデックス
    leader_indices = [i for i, (_, leader, _) in enumerate(employees) if leader == 1]
    # 各社員のスキル
    skills = [emp[2] for emp in employees]

    # リーダー数 < グループ数 の場合は物理的に不可能
    if len(leader_indices) < M:
        st.warning("リーダー数がグループ数より少ないため、各グループにリーダーを配置できません。")
        return [], []

    # 1) 変数定義: x[i][k] = 社員iがグループkに所属 → 1 or 0
    gen = BinarySymbolGenerator()
    x = gen.array(N, M)

    # 2) グループ別スキル合計
    #    S[k] = Σ_i ( skill_i * x[i][k] )
    S = []
    for k in range(M):
        s_k = sum_poly(skills[i] * x[i][k] for i in range(N))
        S.append(s_k)

    # 3) グループ間スキル差(二乗)を全ペアで足し合わせる
    skill_diff_cost = 0
    for k in range(M):
        for l in range(k+1, M):
            diff = S[k] - S[l]
            skill_diff_cost += diff * diff

    # 4) 制約ペナルティ
    # (a) 各社員が exactly 1グループ所属 → sum_k x[i][k] = 1
    PEN_EMP_ONEGROUP = 1000
    employee_penalty = 0
    for i in range(N):
        sum_groups = sum_poly(x[i][k] for k in range(M))
        diff = sum_groups - 1
        employee_penalty += diff * diff
    employee_penalty *= PEN_EMP_ONEGROUP

    # (b) 各グループにリーダーを1人以上 → リーダー不在ペナルティ
    #     リーダー不在 = ∏_{i in leader} (1 - x[i][k]) = 1
    PEN_GROUP_LEADER = 1000
    no_leader_pen = 0
    for k in range(M):
        prod_term = 1
        for i in leader_indices:
            prod_term *= (1 - x[i][k])
        no_leader_pen += prod_term
    no_leader_pen *= PEN_GROUP_LEADER

    # 5) 総コスト
    cost = skill_diff_cost + employee_penalty + no_leader_pen

    # ============= ここで Amplify AE (クラウド) クライアントを設定 =============
    client = FixstarsClient()
    client.token = token  # Amplify AE 管理画面等で取得したトークンを設定
    # タイムアウトや繰り返し回数などパラメータを設定 (任意)
    client.parameters.timeout = 5 * 1000  # ミリ秒 (例: 5秒)
    # client.parameters.num_run = 3       # 実行回数(Amplify AE側での反復)

    solver = Solver(client=client)
    # ===========================================================================

    # 6) ソルバー実行
    result = solver.solve(cost)
    if len(result) == 0:
        st.warning("解が見つかりませんでした。")
        return [], []

    best_values = result[0].values

    # 7) 解釈: x[i][k]=1となっているkを社員iの所属先とみなす
    groups = [[] for _ in range(M)]
    group_skills = [0 for _ in range(M)]
    for i, (emp_id, leader, skill) in enumerate(employees):
        assigned_group = None
        for k in range(M):
            if best_values[x[i][k]] == 1:
                assigned_group = k
                break
        if assigned_group is None:
            assigned_group = 0  # 万が一見つからない場合

        groups[assigned_group].append(emp_id)
        group_skills[assigned_group] += skill

    return groups, group_skills


def main():
    st.title("Amplify AE で社員分割 (スキル均等化)")
    token = 'Your_Amplify_AE_token'

    # CSVアップロード
    uploaded_file = st.file_uploader("社員データCSVをアップロード", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding="shift-jis")
        st.write("読み込んだデータ:")
        st.dataframe(df.head())

        # グループ数指定
        M = st.number_input("いくつのグループに分けますか？", min_value=2, max_value=20, value=2, step=1)

        # 実行ボタン
        if st.button("最適化を実行"):
            # dfを (社員番号, リーダー, スキル) のリストへ
            employees_data = []
            for idx, row in df.iterrows():
                emp_id = int(row["社員番号"])
                leader = int(row["リーダー"])
                skill = int(row["スキル"])
                employees_data.append((emp_id, leader, skill))

            groups, group_skills = solve_skill_partition_multi_group(employees_data, M, token)

            if groups:
                st.subheader("分割結果")
                for k in range(M):
                    st.write(f"**グループ {k}**: 社員番号 {groups[k]}")
                    st.write(f"- スキル合計: {group_skills[k]}")

                max_skill = max(group_skills) if group_skills else 0
                min_skill = min(group_skills) if group_skills else 0
                st.write(f"スキル合計の最大: {max_skill}, 最小: {min_skill}, 差: {max_skill - min_skill}")

    else:
        st.info("CSVファイルをアップロードしてください。")

if __name__ == "__main__":
    main()
