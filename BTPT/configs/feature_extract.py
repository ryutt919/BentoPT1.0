feature_extraction = dict(
    output_dir='data/features',
    feature_type='backbone',  # 'backbone' or 'cls_head'
    save_format='pkl',
    visualize=True,
    visualization=dict(
        method='tsne',  # 'tsne' or 'pca'
        n_components=2,
        perplexity=30.0,
        random_state=42
    )
)