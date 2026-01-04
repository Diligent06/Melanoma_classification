import torch
import matplotlib.pyplot as plt

for item in range(4):
    prob_path = f'./value_tensor_{item}.pt'
    prob_tensor = torch.load(prob_path)
    
    prob = prob_tensor.squeeze(0).cpu().numpy()
    classes = [f"Class {i}" for i in range(8)]

    plt.figure(figsize=(10, 0.5))

    left = 0.0
    colors = plt.cm.tab10.colors

    for i, (p, cls) in enumerate(zip(prob, classes)):
        plt.barh(
            y=0,
            width=p,
            left=left,
            color=colors[i % len(colors)],
            edgecolor="white",
            label=f"{cls}: {p:.2f}"
        )

        # 可选：在条内部标注百分比
        if p > 0.05:
            plt.text(
                left + p / 2,
                0,
                f"{p*100:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=9
            )

        left += p  # ⚠️ 累加必须在循环内

    # ===== 循环结束后，再做这些 =====
    plt.xlim(0, 1)
    plt.yticks([])
    # plt.xlabel("Probability")
    # plt.title("Class Probability Distribution")

    # plt.legend(
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, -0.4),
    #     ncol=4,
    #     frameon=False
    # )

    plt.tight_layout()
    plt.savefig(f"gt_class_prob_stacked_bar_{item}.png", dpi=200, bbox_inches="tight")
    plt.close()
    
    # from IPython import embed; embed()