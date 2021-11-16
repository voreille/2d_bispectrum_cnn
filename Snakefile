names = ["unetlight_default"]

rule all:
    input:
        expand("models/{name}", name=names)

rule train_model:
    input:
        "src/models/monuseg/configs/{name}.yaml"
    output:
        "models/{name}"
    resources:
        gpu_id=1
    shell:
        """
        python src/models/monuseg/train_model.py --config {input} 
        --model_path {output} --gpu_id {resources.gpu_id}
        """