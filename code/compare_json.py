# !/usr/bin/env python3
""" 计算生成 JSON 的准确性 """
import json
import sys
from nltk.translate.bleu_score import sentence_bleu


def compare_string(str1, str2):
    """
    :param str1: 参考字符串
    :param str2: 生成字符串
    :return: 生成字符串与参考字符串的准确性，结果是 0-1 之间
    """
    return sentence_bleu([[*str1]],
                         [*str2], weights=[1])


def compare_json_accuracy(reference, generated, return_raw=False):
    """
    :param reference: 参考 JSON 文件或对象
    :param generated: 生成的 JSON 文件或对象
    :return: 生成的 JSON 文件与参考 JSON 文件的准确性，结果是 0-1 之间
    """
    if isinstance(reference, str):
        reference = json.loads(reference)
    if isinstance(generated, str):
        try:
            generated = json.loads(generated)
        except json.decoder.JSONDecodeError:
            if return_raw:
                return {
                    "missed_keys": ["json error"],
                    "reference_count": 1,
                    "generated_score": 0
                }
            else:
                return 0

    # 避免报错
    if not isinstance(reference, dict):
        reference = {}

    # 计数，之所以用对象是后面传引入
    result = {
        # 没有正确生成的 key
        "missed_keys": [],
        # 参考文件的 key 个数
        "reference_count": 0,
        # 生成文件的得分
        "generated_score": 0
    }
    deep_compare("", reference, generated, result)
    # 单测状态下打印信息方便调试
    if 'unittest' in sys.modules.keys():
        print(result)
    if return_raw:
        return result
    return result["generated_score"] / result["reference_count"]


# 这些 key 必须值一样
hard_keys = set(['type'])

# 这些 key 可以不一样
soft_keys = set(['name', 'description'])


def is_number(num):
    return isinstance(num, int) or isinstance(num, float)


def deep_compare(path_key, reference, generated, result):
    """
     path_key 是用来生成当前路径
    """
    if not isinstance(reference, dict):
        return

    # 不是 dict 说明错了，直接用空对象替代
    if not isinstance(generated, dict):
        generated = {}

    for key in reference:
        result["reference_count"] += 1
        if path_key == "":
            path = key
        else:
            path = f"{path_key}.{key}"
        reference_value = reference[key]

        if key not in generated:
            result["missed_keys"].append(path)
            # 这时还需要继续计算有多少 key
            if isinstance(reference_value, dict):
                deep_compare(path, reference_value, {}, result)
            elif isinstance(reference_value, list):
                for i in range(len(reference_value)):
                    deep_compare(path,
                                 reference_value[i], {}, result)
            continue

        if key in generated:
            generated_value = generated[key]
            # 字符串的情况，直接比较
            if isinstance(reference_value, str) and isinstance(generated_value, str):
                if key in hard_keys:
                    if reference_value == generated_value:
                        result["generated_score"] += 1
                    else:
                        result["missed_keys"].append(path)
                elif key in soft_keys:
                    result["generated_score"] += 1
                else:
                    result["generated_score"] += compare_string(
                        reference_value, generated_value)

            # 数字的情况，直接比较值
            elif is_number(reference_value) and is_number(generated_value):
                if reference_value == generated_value:
                    result["generated_score"] += 1

            elif isinstance(reference_value, list) and isinstance(generated_value, list):
                result["generated_score"] += 1
                generated_value_len = len(generated_value)
                for i in range(len(reference_value)):
                    if i >= generated_value_len:
                        result["missed_keys"].append(path)
                        deep_compare(path + f"[{i}]", reference_value[i], {},
                                     result)
                    else:
                        deep_compare(path + f"[{i}]", reference_value[i], generated_value[i],
                                     result)

            elif isinstance(reference_value, dict) and isinstance(generated_value, dict):
                result["generated_score"] += 1
                deep_compare(path,
                             reference_value, generated_value, result)

            # 其它情况
            elif reference_value == generated_value:
                result["generated_score"] += 1
            else:

                result["missed_keys"].append(path)
