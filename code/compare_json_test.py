""" compare 的单元测试 """
import unittest
from compare_json import compare_json_accuracy


class TestAccuracy(unittest.TestCase):
    """
    测试准确性
    """

    def test_simple(self):
        """
        简单场景
        """
        self.assertEqual(compare_json_accuracy('{"a": 1}', '{"a": 1}'), 1.0)
        # 多生成字段没关系
        self.assertEqual(compare_json_accuracy(
            '{"a": 1}', '{"a": 1, "c": 2}'), 1.0)
        self.assertEqual(compare_json_accuracy('{"a": 1}', '{"a": 2}'), 0.0)
        self.assertEqual(compare_json_accuracy('{"a": 1}', '{"b": 1}'), 0.0)
        self.assertEqual(compare_json_accuracy(
            '{"type": "ab"}', '{"type": "cd"}'), 0.0)
        self.assertEqual(compare_json_accuracy(
            '{"name": "about true"}', '{"name": "true"}'), 1)
        self.assertEqual(compare_json_accuracy(
            '{"a": 1, "d": "about true"}', '{"a": 1, "d": "true"}'), 0.611565080074215)

    def test_deep(self):
        """
        嵌套场景
        """
        self.assertEqual(compare_json_accuracy(
            {
                "a": 1,
                "body": {
                    "type": "input",
                    "name": "user"
                }
            }, {
                "a": 1,
                "controls": {
                    "type": "input",
                    "name": "user"
                }
            }), 0.25)

        self.assertEqual(compare_json_accuracy(
            {
                "a": 1,
                "body": {
                    "type": "input",
                    "value": "user"
                }
            }, {
                "a": 1,
                "body": {
                    "type": "input",
                    "value": "user name"
                }
            }), 0.8611111111111112)

    def test_list(self):
        """
        列表的情况
        """
        self.assertEqual(compare_json_accuracy(
            {
                "body": [{
                    "name": "user"
                }, {
                    "name": "password"
                }]
            }, {
                "body": [{
                    "name": "user"
                }]
            }), 0.6666666666666666)

    def test_boolean(self):
        """
        列表的情况
        """
        self.assertEqual(compare_json_accuracy(
            {
                "select": [{"column": "growth"}],
                "from": "ums_member",
                "distinct": True
            }, {
                "select": [{"column": "growth"}],
                "from": "ums_member",
                "distinct": True
            }), 1.0)


if __name__ == '__main__':
    unittest.main()
