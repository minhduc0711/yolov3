import os


def generate_txt_file(data_dir, output_path):
    with open(output_path, "w") as f:
        for img_name in sorted(os.listdir(data_dir)):
            _, ext = os.path.splitext(img_name)
            if ext.lower() in [".png", ".jpg", ".jpeg"]:
                img_path = os.path.join(data_dir, img_name)
                f.write(img_path + "\n")


generate_txt_file("data/train", "data/corn_train.txt")
generate_txt_file("data/test", "data/corn_test.txt")
