#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import pprint
import re
import sys
import traceback
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, List

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    pd = None
    PANDAS_IMPORT_ERROR = exc
else:
    PANDAS_IMPORT_ERROR = None


UNIT = 10_000
STANDARD_COLUMNS = [
    "状态", "申请金额", "实际支付金额", "预算项目", "资金性质", "部门支出经济分类",
    "拨款金额", "预算管理级次", "项目名称",
]
MUNICIPAL_NATURES = ["111-一般公共预算资金", "121-政府性基金预算资金"]


def round_wan(yuan: float) -> int:
    """元转万元，按常规四舍五入保留整数。"""
    if yuan is None or pd.isna(yuan):
        return 0
    return int((Decimal(str(float(yuan))) / Decimal(UNIT)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def to_amount(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0)


def empty_frame(columns: List[str] | None = None) -> pd.DataFrame:
    return pd.DataFrame(columns=columns or STANDARD_COLUMNS)


def read_excel_auto_header(file: str, required_cols: List[str]) -> pd.DataFrame:
    """自动定位包含 required_cols 的表头行，兼容第一行或第二行为表头。"""
    p = Path(__file__).with_name(file)
    if not p.exists():
        return empty_frame(required_cols)

    raw = pd.read_excel(p, header=None)
    header_idx = None
    for idx, row in raw.iterrows():
        values = {str(v).strip() for v in row.dropna().tolist()}
        if all(col in values for col in required_cols):
            header_idx = idx
            break

    if header_idx is None:
        return empty_frame(required_cols)

    columns = raw.iloc[header_idx].fillna("").map(lambda x: str(x).strip()).tolist()
    df = raw.iloc[header_idx + 1:].copy()
    df.columns = columns
    df = df.loc[:, [c for c in df.columns if c]]

    if "状态" in df.columns:
        status = df["状态"].astype(str).str.strip()
        df = df[df["状态"].notna() & status.ne("")]

    amount_cols = [c for c in ["申请金额", "实际支付金额", "拨款金额"] if c in df.columns]
    if amount_cols:
        amount_mask = False
        for col in amount_cols:
            amount_mask = amount_mask | pd.to_numeric(df[col], errors="coerce").notna()
        df = df[amount_mask]

    return df


def load(file: str) -> pd.DataFrame:
    return read_excel_auto_header(file, ["状态"])


def find_transfer_file(prefix: str) -> str | None:
    files = sorted(Path(__file__).parent.glob(f"{prefix}*.xlsx"))
    return files[0].name if files else None


def load_transfer(prefix: str) -> pd.DataFrame:
    file = find_transfer_file(prefix)
    if not file:
        return empty_frame(["状态", "拨款金额", "资金性质", "预算管理级次", "项目名称"])
    df = read_excel_auto_header(file, ["状态", "拨款金额"])
    if "拨款金额" in df.columns:
        df = df[to_amount(df["拨款金额"]) > 0]
    return df


def transfer_pending_by_level(level: str) -> pd.DataFrame:
    df = load_transfer("实拨待审")
    if df.empty or "预算管理级次" not in df.columns:
        return df.iloc[0:0].copy()
    return df[df["预算管理级次"].eq(level)].copy()


def transfer_approved_by_level(level: str) -> pd.DataFrame:
    df = load_transfer("实拨已审")
    if df.empty or "预算管理级次" not in df.columns:
        return df.iloc[0:0].copy()
    return df[df["预算管理级次"].eq(level)].copy()


def transfer_paid_by_level(level: str) -> pd.DataFrame:
    df = load_transfer("实拨已付")
    if df.empty or "预算管理级次" not in df.columns:
        return df.iloc[0:0].copy()
    return df[df["预算管理级次"].eq(level)].copy()


def transfer_spent_by_level(level: str) -> pd.DataFrame:
    frames = [transfer_approved_by_level(level), transfer_paid_by_level(level)]
    frames = [df for df in frames if not df.empty]
    if not frames:
        return empty_frame(["状态", "拨款金额", "资金性质", "预算管理级次", "项目名称"])
    return pd.concat(frames, ignore_index=True)


def report_date_str() -> str:
    for prefix in ("实拨待审", "实拨已审", "实拨已付"):
        file = find_transfer_file(prefix)
        if not file:
            continue
        match = re.search(r"(\d{1,2})\.(\d{1,2})", file)
        if match:
            return f"{int(match.group(1))}月{int(match.group(2))}日"

    today = datetime.now()
    return f"{today.month}月{today.day}日"


def sum_yuan(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df.columns:
        return 0.0
    return float(to_amount(df[col]).sum())


def sum_round(df: pd.DataFrame, col: str) -> int:
    return round_wan(sum_yuan(df, col))


def group_sum_yuan(df: pd.DataFrame, group_col: str, sum_col: str) -> pd.Series:
    if df.empty or group_col not in df.columns or sum_col not in df.columns:
        return pd.Series([], dtype=float)
    return to_amount(df[sum_col]).groupby(df[group_col]).sum()


def add_series_yuan(*series_list: pd.Series) -> pd.Series:
    result = pd.Series([], dtype=float)
    for series in series_list:
        if series is None or series.empty:
            continue
        result = result.add(series, fill_value=0)
    return result


def project_dict_from_yuan(series: pd.Series) -> Dict[str, int]:
    if series.empty:
        return {}
    rounded = series.apply(round_wan)
    for threshold in (20, 10, 5):
        filtered = rounded[rounded >= threshold]
        if not filtered.empty:
            return filtered.astype(int).to_dict()
    return {}


def normalize_voucher(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def voucher_set(df: pd.DataFrame) -> set[str]:
    if df.empty or "支付凭证号" not in df.columns:
        return set()
    return {value for value in df["支付凭证号"].map(normalize_voucher) if value}


def remove_overlapping_voucher_rows(
    yest_df: pd.DataFrame,
    paid_df: pd.DataFrame,
    pend_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """昨天凭证号若已出现在已支付或今天已审未支付，则三张表中同凭证号行都不参与支出计算。"""
    if yest_df.empty or "支付凭证号" not in yest_df.columns:
        return paid_df, yest_df, pend_df

    duplicate_vouchers = voucher_set(yest_df) & (voucher_set(paid_df) | voucher_set(pend_df))
    if not duplicate_vouchers:
        return paid_df, yest_df, pend_df

    def remove_from(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "支付凭证号" not in df.columns:
            return df
        vouchers = df["支付凭证号"].map(normalize_voucher)
        keep_mask = vouchers.eq("") | ~vouchers.isin(duplicate_vouchers)
        return df[keep_mask].copy()

    return remove_from(paid_df), remove_from(yest_df), remove_from(pend_df)


def calculate_total_yuan_with_adjustment(
    paid_df: pd.DataFrame,
    yest_df: pd.DataFrame,
    pend_df: pd.DataFrame,
    paid_col: str,
    pending_col: str,
    filter_municipal: bool = False,
) -> float:
    if filter_municipal:
        paid_df = paid_df[paid_df["资金性质"].isin(MUNICIPAL_NATURES)] if "资金性质" in paid_df.columns else paid_df
        yest_df = yest_df[yest_df["资金性质"].isin(MUNICIPAL_NATURES)] if "资金性质" in yest_df.columns else yest_df
        pend_df = pend_df[pend_df["资金性质"].isin(MUNICIPAL_NATURES)] if "资金性质" in pend_df.columns else pend_df

    paid_by_proj = group_sum_yuan(paid_df, "预算项目", paid_col)
    yest_by_proj = group_sum_yuan(yest_df, "预算项目", pending_col)
    pend_by_proj = group_sum_yuan(pend_df, "预算项目", pending_col)

    all_projects = paid_by_proj.index.union(yest_by_proj.index).union(pend_by_proj.index)
    paid_by_proj = paid_by_proj.reindex(all_projects, fill_value=0)
    yest_by_proj = yest_by_proj.reindex(all_projects, fill_value=0)
    pend_by_proj = pend_by_proj.reindex(all_projects, fill_value=0)

    base_total = paid_by_proj.sum() - yest_by_proj.sum() + pend_by_proj.sum()
    adjustment = (yest_by_proj - paid_by_proj).apply(lambda x: max(x, 0)).sum()
    return float(base_total + adjustment)


def calculate_total_with_adjustment(
    paid_df: pd.DataFrame,
    yest_df: pd.DataFrame,
    pend_df: pd.DataFrame,
    paid_col: str,
    pending_col: str,
    filter_municipal: bool = False,
) -> int:
    return round_wan(calculate_total_yuan_with_adjustment(
        paid_df, yest_df, pend_df, paid_col, pending_col, filter_municipal
    ))


def calc_sanbao_expense() -> int:
    paid = load("支出情况-三保-已支付.xlsx")
    yest_pending = load("支出情况-三保-昨天已审未支付.xlsx")
    pending = load("支出情况-三保-已审未支付.xlsx")
    return sum_round(paid, "实际支付金额") - sum_round(yest_pending, "申请金额") + sum_round(pending, "申请金额")


def last_year_carry_total() -> int:
    paid = load("支出情况-上年结转上级资金-已支付.xlsx")
    yest = load("支出情况-上年结转上级资金-昨天已审未支付.xlsx")
    pend = load("支出情况-上年结转上级资金-已审未支付.xlsx")
    paid, yest, pend = remove_overlapping_voucher_rows(yest, paid, pend)
    total = calculate_total_yuan_with_adjustment(paid, yest, pend, "实际支付金额", "申请金额")
    total += sum_yuan(transfer_spent_by_level("上年结转上级"), "拨款金额")
    return round_wan(total)


def last_year_carry_expense() -> Dict[str, int]:
    paid_df = load("支出情况-上年结转上级资金-已支付.xlsx")
    yest_df = load("支出情况-上年结转上级资金-昨天已审未支付.xlsx")
    pend_df = load("支出情况-上年结转上级资金-已审未支付.xlsx")
    paid_df, yest_df, pend_df = remove_overlapping_voucher_rows(yest_df, paid_df, pend_df)
    paid = group_sum_yuan(paid_df, "预算项目", "实际支付金额")
    yest = group_sum_yuan(yest_df, "预算项目", "申请金额")
    pend = group_sum_yuan(pend_df, "预算项目", "申请金额")
    transfer = group_sum_yuan(transfer_spent_by_level("上年结转上级"), "项目名称", "拨款金额")
    return project_dict_from_yuan(paid.subtract(yest, fill_value=0).add(pend, fill_value=0).add(transfer, fill_value=0))


def current_year_upper_total() -> int:
    paid = load("支出情况-当年度上级资金-已支付.xlsx")
    yest = load("支出情况-当年度上级资金-昨天已审未支付.xlsx")
    pend = load("支出情况-当年度上级资金-已审未支付.xlsx")
    paid, yest, pend = remove_overlapping_voucher_rows(yest, paid, pend)
    total = calculate_total_yuan_with_adjustment(paid, yest, pend, "实际支付金额", "申请金额")
    total += sum_yuan(transfer_spent_by_level("本年度上级"), "拨款金额")
    return round_wan(total)


def current_year_upper_expense() -> Dict[str, int]:
    paid_df = load("支出情况-当年度上级资金-已支付.xlsx")
    yest_df = load("支出情况-当年度上级资金-昨天已审未支付.xlsx")
    pend_df = load("支出情况-当年度上级资金-已审未支付.xlsx")
    paid_df, yest_df, pend_df = remove_overlapping_voucher_rows(yest_df, paid_df, pend_df)
    paid = group_sum_yuan(paid_df, "预算项目", "实际支付金额")
    yest = group_sum_yuan(yest_df, "预算项目", "申请金额")
    pend = group_sum_yuan(pend_df, "预算项目", "申请金额")
    transfer = group_sum_yuan(transfer_spent_by_level("本年度上级"), "项目名称", "拨款金额")
    return project_dict_from_yuan(paid.subtract(yest, fill_value=0).add(pend, fill_value=0).add(transfer, fill_value=0))


def _expense_by_nature(natures: List[str], amount_col: str, file: str) -> pd.Series:
    df = load(file)
    if df.empty or "资金性质" not in df.columns:
        return pd.Series([], dtype=float)
    return group_sum_yuan(df[df["资金性质"].isin(natures)], "预算项目", amount_col)


def expense_frames_by_nature(natures: List[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paid = load("支出情况-市级和债券-已支付.xlsx")
    yest = load("支出情况-市级和债券-昨天已审未支付.xlsx")
    pend = load("支出情况-市级和债券-已审未支付.xlsx")

    def filter_nature(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "资金性质" not in df.columns:
            return df
        return df[df["资金性质"].isin(natures)].copy()

    paid = filter_nature(paid)
    yest = filter_nature(yest)
    pend = filter_nature(pend)
    paid, yest, pend = remove_overlapping_voucher_rows(yest, paid, pend)
    return paid, yest, pend


def municipal_total() -> int:
    paid, yest, pend = expense_frames_by_nature(MUNICIPAL_NATURES)
    total = calculate_total_yuan_with_adjustment(paid, yest, pend, "实际支付金额", "申请金额")
    total += sum_yuan(transfer_spent_by_level("市级"), "拨款金额")
    return round_wan(total)


def municipal_expense() -> Dict[str, int]:
    paid_df, yest_df, pend_df = expense_frames_by_nature(MUNICIPAL_NATURES)
    paid = group_sum_yuan(paid_df, "预算项目", "实际支付金额")
    yest = group_sum_yuan(yest_df, "预算项目", "申请金额")
    pend = group_sum_yuan(pend_df, "预算项目", "申请金额")
    base_projects = project_dict_from_yuan(paid.subtract(yest, fill_value=0).add(pend, fill_value=0))
    transfer_projects = project_dict_from_yuan(group_sum_yuan(transfer_spent_by_level("市级"), "项目名称", "拨款金额"))
    return {**base_projects, **transfer_projects}


def general_bond_total() -> int:
    paid, yest, pend = expense_frames_by_nature(["112-一般债券"])
    total = calculate_total_yuan_with_adjustment(paid, yest, pend, "实际支付金额", "申请金额")
    total += sum_yuan(transfer_spent_by_level("一般债券"), "拨款金额")
    return round_wan(total)


def general_bond_expense() -> Dict[str, int]:
    paid_df, yest_df, pend_df = expense_frames_by_nature(["112-一般债券"])
    paid = group_sum_yuan(paid_df, "预算项目", "实际支付金额")
    yest = group_sum_yuan(yest_df, "预算项目", "申请金额")
    pend = group_sum_yuan(pend_df, "预算项目", "申请金额")
    transfer = group_sum_yuan(transfer_spent_by_level("一般债券"), "项目名称", "拨款金额")
    return project_dict_from_yuan(paid.subtract(yest, fill_value=0).add(pend, fill_value=0).add(transfer, fill_value=0))


def special_bond_total() -> int:
    paid, yest, pend = expense_frames_by_nature(["122-专项债券"])
    total = calculate_total_yuan_with_adjustment(paid, yest, pend, "实际支付金额", "申请金额")
    total += sum_yuan(transfer_spent_by_level("专项债券"), "拨款金额")
    return round_wan(total)


def special_bond_expense() -> Dict[str, int]:
    paid_df, yest_df, pend_df = expense_frames_by_nature(["122-专项债券"])
    paid = group_sum_yuan(paid_df, "预算项目", "实际支付金额")
    yest = group_sum_yuan(yest_df, "预算项目", "申请金额")
    pend = group_sum_yuan(pend_df, "预算项目", "申请金额")
    transfer = group_sum_yuan(transfer_spent_by_level("专项债券"), "项目名称", "拨款金额")
    return project_dict_from_yuan(paid.subtract(yest, fill_value=0).add(pend, fill_value=0).add(transfer, fill_value=0))


def pending_total() -> int:
    files = [
        "待拨付情况-三保.xlsx",
        "待拨付情况-上年结转上级资金.xlsx",
        "待拨付情况-当年度上级资金.xlsx",
        "待拨付情况-市级和债券.xlsx",
    ]
    total = sum(sum_yuan(load(file), "申请金额") for file in files)
    total += sum_yuan(load_transfer("实拨待审"), "拨款金额")
    return round_wan(total)


def pending_sanbao() -> int:
    total = sum_yuan(load("待拨付情况-三保.xlsx"), "申请金额")
    total += sum_yuan(transfer_pending_by_level("三保"), "拨款金额")
    return round_wan(total)


def pending_upper() -> int:
    total = sum_yuan(load("待拨付情况-上年结转上级资金.xlsx"), "申请金额")
    total += sum_yuan(load("待拨付情况-当年度上级资金.xlsx"), "申请金额")
    total += sum_yuan(transfer_pending_by_level("上年结转上级"), "拨款金额")
    total += sum_yuan(transfer_pending_by_level("本年度上级"), "拨款金额")
    return round_wan(total)


def pending_last_year_total() -> int:
    total = sum_yuan(load("待拨付情况-上年结转上级资金.xlsx"), "申请金额")
    total += sum_yuan(transfer_pending_by_level("上年结转上级"), "拨款金额")
    return round_wan(total)


def pending_last_year_carry() -> Dict[str, int]:
    base = group_sum_yuan(load("待拨付情况-上年结转上级资金.xlsx"), "预算项目", "申请金额")
    transfer = group_sum_yuan(transfer_pending_by_level("上年结转上级"), "项目名称", "拨款金额")
    return project_dict_from_yuan(add_series_yuan(base, transfer))


def pending_current_upper_total() -> int:
    total = sum_yuan(load("待拨付情况-当年度上级资金.xlsx"), "申请金额")
    total += sum_yuan(transfer_pending_by_level("本年度上级"), "拨款金额")
    return round_wan(total)


def pending_current_upper() -> Dict[str, int]:
    base = group_sum_yuan(load("待拨付情况-当年度上级资金.xlsx"), "预算项目", "申请金额")
    transfer = group_sum_yuan(transfer_pending_by_level("本年度上级"), "项目名称", "拨款金额")
    return project_dict_from_yuan(add_series_yuan(base, transfer))


def pending_general_bond_total() -> int:
    df = load("待拨付情况-市级和债券.xlsx")
    base = df[df["资金性质"].eq("112-一般债券")] if "资金性质" in df.columns else df
    total = sum_yuan(base, "申请金额")
    total += sum_yuan(transfer_pending_by_level("一般债券"), "拨款金额")
    return round_wan(total)


def pending_general_bond() -> Dict[str, int]:
    df = load("待拨付情况-市级和债券.xlsx")
    base_df = df[df["资金性质"].eq("112-一般债券")] if "资金性质" in df.columns else df
    base = group_sum_yuan(base_df, "预算项目", "申请金额")
    transfer = group_sum_yuan(transfer_pending_by_level("一般债券"), "项目名称", "拨款金额")
    return project_dict_from_yuan(add_series_yuan(base, transfer))


def pending_special_bond_total() -> int:
    df = load("待拨付情况-市级和债券.xlsx")
    base = df[df["资金性质"].eq("122-专项债券")] if "资金性质" in df.columns else df
    total = sum_yuan(base, "申请金额")
    total += sum_yuan(transfer_pending_by_level("专项债券"), "拨款金额")
    return round_wan(total)


def pending_special_bond() -> Dict[str, int]:
    df = load("待拨付情况-市级和债券.xlsx")
    base_df = df[df["资金性质"].eq("122-专项债券")] if "资金性质" in df.columns else df
    base = group_sum_yuan(base_df, "预算项目", "申请金额")
    transfer = group_sum_yuan(transfer_pending_by_level("专项债券"), "项目名称", "拨款金额")
    return project_dict_from_yuan(add_series_yuan(base, transfer))


def pending_municipal_breakdown() -> Dict[str, int]:
    df = load("待拨付情况-市级和债券.xlsx")
    if df.empty or "资金性质" not in df.columns:
        return {}

    df_municipal = df[df["资金性质"].isin(MUNICIPAL_NATURES)].copy()
    total = round_wan(sum_yuan(df_municipal, "申请金额"))
    pub = round_wan(sum_yuan(df[df["资金性质"].eq("111-一般公共预算资金")], "申请金额"))
    gov = round_wan(sum_yuan(df[df["资金性质"].eq("121-政府性基金预算资金")], "申请金额"))

    def pick_contain(keywords: List[str]) -> int:
        if df_municipal.empty or "部门支出经济分类" not in df_municipal.columns:
            return 0
        m = df_municipal["部门支出经济分类"].apply(
            lambda x: any(kw in str(x) for kw in keywords) if pd.notna(x) else False
        )
        return round_wan(sum_yuan(df_municipal.loc[m], "申请金额"))

    personnel = pick_contain(["301", "303"])
    exclude_codes = (
        "301", "303", "委托业务费", "专用设备购置", "办公设备购置",
        "基础设施建设", "大型修缮", "信息网络及软件购置更新",
        "安置补助", "地上附着物和青苗补偿", "拆迁补偿", "其他资本性支出",
    )
    exclude_mask = df_municipal["部门支出经济分类"].apply(
        lambda x: (pd.isna(x) or str(x).strip() == "") or any(code in str(x) for code in exclude_codes)
    )
    operation = round_wan(sum_yuan(df_municipal.loc[~exclude_mask], "申请金额"))
    third = pick_contain(["委托业务费", "专用设备购置", "办公设备购置", "信息网络及软件购置更新"])
    infra = pick_contain(["基础设施建设", "大型修缮", "其他资本性支出"])
    land = pick_contain(["安置补助", "地上附着物和青苗补偿", "拆迁补偿"])

    return {
        "总额": total,
        "一般公共": pub,
        "政府性基金": gov,
        "人员类": personnel,
        "运转类": operation,
        "委托及采购": third,
        "基建类": infra,
        "征地类": land,
    }


def pending_municipal_transfer_breakdown() -> Dict[str, int]:
    df = transfer_pending_by_level("市级")
    if df.empty:
        return {}

    total = round_wan(sum_yuan(df, "拨款金额"))
    pub = round_wan(sum_yuan(df[df["资金性质"].eq("111-一般公共预算资金")], "拨款金额"))
    gov = round_wan(sum_yuan(df[df["资金性质"].eq("121-政府性基金预算资金")], "拨款金额"))
    projects = project_dict_from_yuan(group_sum_yuan(df, "项目名称", "拨款金额"))

    return {
        "总额": total,
        "一般公共": pub,
        "政府性基金": gov,
        "项目": projects,
    }


def clean_zero(obj):
    if isinstance(obj, dict):
        out = {k: clean_zero(v) for k, v in obj.items() if v is not None}
        return {k: v for k, v in out.items() if not isinstance(v, (int, float)) or v > 0}
    if isinstance(obj, list):
        return [clean_zero(i) for i in obj if i is not None]
    return obj


def clean_below_20(obj):
    if not isinstance(obj, dict):
        return obj

    result = {}
    for section_name, section_content in obj.items():
        if not isinstance(section_content, dict):
            result[section_name] = section_content
            continue

        total_keys = [k for k in section_content.keys() if "-总额（万元）" in k]
        keys_to_remove = set()
        for total_key in total_keys:
            total_value = section_content.get(total_key, 0)
            if isinstance(total_value, (int, float)) and total_value < 20:
                prefix = total_key.split("-总额（万元）")[0]
                keys_to_remove.add(total_key)
                keys_to_remove.add(f"{prefix}（项目明细）")

        result[section_name] = {k: v for k, v in section_content.items() if k not in keys_to_remove}
    return result


def should_show_mainly(series: dict, total: int) -> bool:
    return bool(series) and sum(series.values()) != total


def project_name(name) -> str:
    return str(name).split("-", 1)[-1]


def generate_narrative(d: dict) -> str:
    date_str = report_date_str()

    def top(series: dict, n: int = 100, compact: bool = False) -> str:
        items = sorted(series.items(), key=lambda x: -x[1])[:n]
        if compact:
            return "，".join(f"{project_name(k)}{v}万元" for k, v in items)
        return "，".join(f"{project_name(k)} {v} 万元" for k, v in items)

    exp = d.get("一、支出情况", {})
    sanbao = max(exp.get("1 三保资金（万元）", 0), 0)
    last_up = max(exp.get("2 上年结转上级资金-总额（万元）", 0), 0)
    curr_up = max(exp.get("3 当年度上级资金-总额（万元）", 0), 0)
    municipal = max(exp.get("4 市级资金-总额（万元）", 0), 0)
    bond = max(exp.get("5 一般债券资金-总额（万元）", 0), 0)
    special_bond = max(exp.get("6 专项债券资金-总额（万元）", 0), 0)

    upper_total = last_up + curr_up
    total_spent = sanbao + upper_total + municipal + bond + special_bond

    lines = ["三、支出情况", f"{date_str}财政复核已审银行已支出{total_spent}万元："]
    seq = 0

    if sanbao > 0:
        seq += 1
        lines.append(f"一是三保资金 {sanbao} 万元；")

    if upper_total > 0:
        seq += 1
        sub_items_data = []
        if last_up > 0:
            details = exp.get("2 上年结转上级资金（项目明细）", {})
            sub_items_data.append(("上年结转上级资金", last_up, details, should_show_mainly(details, last_up)))
        if curr_up > 0:
            details = exp.get("3 当年度上级资金（项目明细）", {})
            sub_items_data.append(("当年度上级资金", curr_up, details, should_show_mainly(details, curr_up)))

        seq_char = ["一", "二", "三", "四", "五"][seq - 1] if seq <= 5 else "五"
        if len(sub_items_data) == 1:
            name, amount, details, show_mainly = sub_items_data[0]
            details_text = ""
            if details:
                details_text = f"，主要为：{top(details, compact=True)}" if show_mainly else f"：{top(details, compact=True)}"
            lines.append(f"{seq_char}是上级资金{upper_total}万元，为{name}{amount}万元{details_text}")
        else:
            sub_items = []
            for idx, (name, amount, details, show_mainly) in enumerate(sub_items_data):
                details_text = ""
                if details:
                    details_text = f"，主要为：{top(details)}" if show_mainly else f"：{top(details)}"
                sub_items.append(f"{idx + 1}. {name} {amount} 万元{details_text}")
            lines.append(f"{seq_char}是上级资金 {upper_total} 万元：" + "；".join(sub_items) + "；")

    if municipal > 0:
        seq += 1
        details = exp.get("4 市级资金（项目明细）", {})
        seq_char = ["一", "二", "三", "四", "五"][seq - 1] if seq <= 5 else "五"
        lines.append(f"{seq_char}是市级资金 {municipal} 万元"
                     + (f"，主要为：{top(details)}" if should_show_mainly(details, municipal) and details else
                        f"：{top(details)}" if details else "") + "；")

    if bond > 0:
        seq += 1
        details = exp.get("5 一般债券资金（项目明细）", {})
        seq_char = ["一", "二", "三", "四", "五"][seq - 1] if seq <= 5 else "五"
        lines.append(f"{seq_char}是一般债券资金 {bond} 万元"
                     + (f"，主要为：{top(details)}" if should_show_mainly(details, bond) and details else
                        f"：{top(details)}" if details else "") + "；")

    if special_bond > 0:
        seq += 1
        details = exp.get("6 专项债券资金（项目明细）", {})
        seq_char = ["一", "二", "三", "四", "五"][seq - 1] if seq <= 5 else "五"
        lines.append(f"{seq_char}是专项债券资金 {special_bond} 万元"
                     + (f"，主要为：{top(details)}" if should_show_mainly(details, special_bond) and details else
                        f"：{top(details)}" if details else "") + "；")

    lines.append("四、待拨付情况")
    pend = d.get("二、待拨付情况", {})
    total_p = pend.get("1 待拨付总额（万元）", 0)
    if total_p > 0:
        lines.append(f"截至 {date_str} 待拨付资金 {total_p} 万元")

    pending_seq = 0
    pending_seq_chars = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]

    def next_pending_seq() -> str:
        nonlocal pending_seq
        pending_seq += 1
        if pending_seq <= len(pending_seq_chars):
            return pending_seq_chars[pending_seq - 1]
        return str(pending_seq)

    sanbao_p = pend.get("2 三保资金（万元）", 0)
    if sanbao_p > 0:
        lines.append(f"（{next_pending_seq()}）三保资金 {sanbao_p} 万元")

    upper_total_p = pend.get("3 上级资金（万元）", 0)
    if upper_total_p > 0:
        lines.append(f"（{next_pending_seq()}）上级资金 {upper_total_p} 万元")
        sub_items = []
        last_p_total = pend.get("4 上年结转上级资金-总额（万元）", 0)
        curr_p_total = pend.get("5 当年度上级资金-总额（万元）", 0)
        if last_p_total > 0:
            details = pend.get("4 上年结转上级资金（项目明细）", {})
            sub_items.append(("上年结转上级资金", last_p_total, details, should_show_mainly(details, last_p_total)))
        if curr_p_total > 0:
            details = pend.get("5 当年度上级资金（项目明细）", {})
            sub_items.append(("当年度上级资金", curr_p_total, details, should_show_mainly(details, curr_p_total)))

        for idx, (name, amount, details, show_mainly) in enumerate(sub_items):
            details_text = ""
            if details:
                details_text = f"，主要为：{top(details)}" if show_mainly else f"：{top(details)}"
            prefix = f"{idx + 1}. " if len(sub_items) > 1 else ""
            lines.append(f"{prefix}{name} {amount} 万元{details_text}；")

    bond_p_total = pend.get("6 一般债券资金-总额（万元）", 0)
    bond_p_big = pend.get("6 一般债券资金（项目明细）", {})
    if bond_p_total > 0:
        lines.append(f"（{next_pending_seq()}）一般债券资金 {bond_p_total} 万元"
                     + (f"，主要为：{top(bond_p_big)}" if should_show_mainly(bond_p_big, bond_p_total) and bond_p_big else
                        f"：{top(bond_p_big)}" if bond_p_big else "") + "；")

    special_bond_total = pend.get("7 专项债券资金-总额（万元）", 0)
    special_bond_big = pend.get("7 专项债券资金（项目明细）", {})
    if special_bond_total > 0:
        lines.append(f"（{next_pending_seq()}）专项债券资金 {special_bond_total} 万元"
                     + (f"，主要为：{top(special_bond_big)}" if should_show_mainly(special_bond_big, special_bond_total) and special_bond_big else
                        f"：{top(special_bond_big)}" if special_bond_big else "") + "；")

    municipal_p = pend.get("8 市级资金细分（万元）", {})
    if municipal_p and municipal_p.get("总额", 0) > 0:
        pub = municipal_p.get("一般公共", 0)
        gov = municipal_p.get("政府性基金", 0)
        lines.append(f"（{next_pending_seq()}）市级资金 {municipal_p['总额']} 万元（一般公共预算 {pub} 万元、政府性基金 {gov} 万元），"
                     f"其中人员类支出 {municipal_p.get('人员类', 0)} 万元，"
                     f"运转类支出 {municipal_p.get('运转类', 0)} 万元，"
                     f"委托第三方服务及采购类支出 {municipal_p.get('委托及采购', 0)} 万元，"
                     f"基建类支出 {municipal_p.get('基建类', 0)} 万元，"
                     f"征地类支出 {municipal_p.get('征地类', 0)} 万元。")

    municipal_transfer = pend.get("9 市级资金（实拨）（万元）", {})
    if municipal_transfer and municipal_transfer.get("总额", 0) > 0:
        pub = municipal_transfer.get("一般公共", 0)
        gov = municipal_transfer.get("政府性基金", 0)
        projects = municipal_transfer.get("项目", {})
        lines.append(f"（{next_pending_seq()}）市级资金（实拨）{municipal_transfer['总额']}万元"
                     f"（一般公共预算资金 {pub}万元、政府性基金{gov}万元）"
                     + (f"，主要为：{top(projects, compact=True)}" if projects else "") + "；")

    return "\n".join(lines)


def build_report() -> dict:
    return {
        "一、支出情况": {
            "1 三保资金（万元）": calc_sanbao_expense(),
            "2 上年结转上级资金-总额（万元）": last_year_carry_total(),
            "2 上年结转上级资金（项目明细）": last_year_carry_expense(),
            "3 当年度上级资金-总额（万元）": current_year_upper_total(),
            "3 当年度上级资金（项目明细）": current_year_upper_expense(),
            "4 市级资金-总额（万元）": municipal_total(),
            "4 市级资金（项目明细）": municipal_expense(),
            "5 一般债券资金-总额（万元）": general_bond_total(),
            "5 一般债券资金（项目明细）": general_bond_expense(),
            "6 专项债券资金-总额（万元）": special_bond_total(),
            "6 专项债券资金（项目明细）": special_bond_expense(),
        },
        "二、待拨付情况": {
            "1 待拨付总额（万元）": pending_total(),
            "2 三保资金（万元）": pending_sanbao(),
            "3 上级资金（万元）": pending_upper(),
            "4 上年结转上级资金-总额（万元）": pending_last_year_total(),
            "4 上年结转上级资金（项目明细）": pending_last_year_carry(),
            "5 当年度上级资金-总额（万元）": pending_current_upper_total(),
            "5 当年度上级资金（项目明细）": pending_current_upper(),
            "6 一般债券资金-总额（万元）": pending_general_bond_total(),
            "6 一般债券资金（项目明细）": pending_general_bond(),
            "7 专项债券资金-总额（万元）": pending_special_bond_total(),
            "7 专项债券资金（项目明细）": pending_special_bond(),
            "8 市级资金细分（万元）": pending_municipal_breakdown(),
            "9 市级资金（实拨）（万元）": pending_municipal_transfer_breakdown(),
        },
    }


def main() -> tuple[dict, str, str]:
    report = build_report()
    report_text = pprint.pformat(report, width=120)
    cleaned = clean_below_20(clean_zero(report))
    narrative = generate_narrative(cleaned)
    return report, report_text, narrative


def write_outputs(report_text: str, narrative: str) -> None:
    out_dir = Path(__file__).parent
    (out_dir / "输出结果.txt").write_text(report_text + "\n\n" + narrative + "\n", encoding="utf-8-sig")
    (out_dir / "总结输出.txt").write_text(narrative + "\n", encoding="utf-8-sig")


def write_error_log(message: str) -> None:
    Path(__file__).with_name("错误日志.txt").write_text(message, encoding="utf-8-sig")


def move_files_after_processing() -> None:
    import shutil

    tomorrow = datetime.now() + timedelta(days=1)
    target_dir = Path(__file__).parent.parent / tomorrow.strftime("%m.%d")
    target_dir.mkdir(parents=True, exist_ok=True)

    rename_map = {
        "支出情况-当年度上级资金-已审未支付.xlsx": "支出情况-当年度上级资金-昨天已审未支付.xlsx",
        "支出情况-上年结转上级资金-已审未支付.xlsx": "支出情况-上年结转上级资金-昨天已审未支付.xlsx",
        "支出情况-三保-已审未支付.xlsx": "支出情况-三保-昨天已审未支付.xlsx",
        "支出情况-市级和债券-已审未支付.xlsx": "支出情况-市级和债券-昨天已审未支付.xlsx",
        "gov_expenditure.py": "gov_expenditure.py",
    }

    for src_filename, dst_filename in rename_map.items():
        src_path = Path(__file__).parent / src_filename
        dst_path = target_dir / dst_filename
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            print(f"已处理: {src_filename} -> {dst_path}")
        else:
            print(f"源文件不存在，跳过: {src_filename}")


def run_cli() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="生成财政支出和待拨付情况总结")
    parser.add_argument("--move", action="store_true", help="处理完成后复制已审未支付文件到明天日期目录")
    args = parser.parse_args()

    if PANDAS_IMPORT_ERROR is not None:
        message = (
            "缺少依赖 pandas/openpyxl，无法读取 Excel。\n"
            "请在命令行执行：pip install pandas openpyxl\n\n"
            f"{PANDAS_IMPORT_ERROR}\n"
        )
        print(message)
        write_error_log(message)
        return 1

    try:
        _, report_text, narrative = main()
        write_outputs(report_text, narrative)
        print(report_text)
        print(narrative)
        print("\n已生成：输出结果.txt、总结输出.txt")
        if args.move:
            move_files_after_processing()
        return 0
    except Exception:
        message = traceback.format_exc()
        print(message)
        write_error_log(message)
        return 1


if __name__ == "__main__":
    raise SystemExit(run_cli())
