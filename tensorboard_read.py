# import os
# from tensorboard.backend.event_processing import event_accumulator

# No = 37

# # 设置日志目录
# logdir = f"ViT_torch/logs/cifar10_No_{No}"  # 替换为你的日志目录

# # 自动找到唯一的 event 文件
# event_file = [f for f in os.listdir(logdir) if "events.out.tfevents" in f]
# if len(event_file) != 1:
#     raise ValueError(f"在目录 {logdir} 下找到了 {len(event_file)} 个 event 文件，请确认！")
# event_file_path = os.path.join(logdir, event_file[0])

# # 加载日志数据
# ea = event_accumulator.EventAccumulator(event_file_path)
# ea.Reload()

# # 打印所有标量名称
# scalars = ea.Tags()["scalars"]
# print(f"所有记录的标量名称：{scalars}")

# # 提取某个标量的统计信息
# scalar_name = "top1_accuracy"  # 替换为你想要分析的标量名称
# scalar_data = ea.Scalars(scalar_name)
# values = [scalar.value for scalar in scalar_data]
# max_value = max(values)

# print(f"标量 '{scalar_name}' 的统计数据：")
# print(f"Top1最大值：{max_value}")

# # 提取某个标量的统计信息
# scalar_name = "top5_accuracy"  # 替换为你想要分析的标量名称
# scalar_data = ea.Scalars(scalar_name)
# values = [scalar.value for scalar in scalar_data]
# max_value = max(values)

# print(f"标量 '{scalar_name}' 的统计数据：")
# print(f"Top5最大值：{max_value}")


import os
from tensorboard.backend.event_processing import event_accumulator

def extract_text_value(logdir, tag):
    """
    Extract the text value associated with a given tag from a TensorBoard log directory.

    Args:
        logdir (str): Path to the TensorBoard log directory.
        tag (str): The tag of the text value to extract.

    Returns:
        list: A list of extracted text values for the given tag.
    """
    # Automatically find the unique event file
    event_file = [f for f in os.listdir(logdir) if "events.out.tfevents" in f]
    if len(event_file) != 1:
        raise ValueError(f"In directory {logdir}, found {len(event_file)} event files. Please check!")
    event_file_path = os.path.join(logdir, event_file[0])

    # Load the event data
    ea = event_accumulator.EventAccumulator(event_file_path)
    ea.Reload()

    # Check if the tag exists under 'tensors'
    if tag not in ea.Tags()["tensors"]:
        raise ValueError(f"Tag '{tag}' not found in the TensorBoard logs.")

    # Extract and decode the text values
    tensor_data = ea.Tensors(tag)
    text_values = [t.tensor_proto.string_val[0].decode("utf-8") for t in tensor_data]

    return text_values

if __name__ == "__main__":
    No = 37

    # Set the log directory
    logdir = f"ViT_torch/logs/cifar10_No_{No}"  
    text_tag = "Configs/Num_Params/text_summary"  

    try:
        text_values = extract_text_value(logdir, text_tag)
        print(f"Extracted text values for tag '{text_tag}':")
        for value in text_values:
            print(value)
    except Exception as e:
        print(f"Error: {e}")
