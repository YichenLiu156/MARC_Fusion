import json


# class GetArgs():
#     def __init__(self, json_path):
#         with open(json_path) as f:
#             args = json.load(f)
#             self.__dict__.update(args)

class GetArgs:
    """
    支持主配置文件 + 网络配置文件合并。

    例如 material_train.json 中包含：
        "network_config_path": "../params_marc/default/fusion_network.json"

    最终 args 会同时拥有：
        训练参数 + 网络结构参数
    """

    def __init__(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            train_args = json.load(f)

        network_config_path = train_args.get("network_config_path", "")

        merged_args = {}

        if network_config_path:
            with open(network_config_path, "r", encoding="utf-8") as f:
                network_args = json.load(f)

            merged_args.update(network_args)

        # 训练参数后更新，优先级高于网络参数
        merged_args.update(train_args)

        self.__dict__.update(merged_args)
if __name__ == '__main__':
    args = GetArgs("../params/default/fusion_network.json")
    print(args.save_weight_path)