import seaborn as sns


class UnivariatePlots:
    @classmethod
    def univariate_distribution_feat_vs_target(cls, data, feat, target, kind="hist", bins=10, ax=None):
        if ax:
            sns.displot(data=data, x=feat, hue=target, kind=kind, bins=bins, ax=ax)
        else:
            sns.displot(data=data, x=feat, hue=target, kind=kind, bins=bins, ax=ax)

    @classmethod
    def numerical_feat_vs_target(cls, data, feat, target, bins=10, stat="probability", ax=None):
        if ax is not None:
            sns.histplot(data=data, x=feat, hue=target, stat=stat, bins=bins, ax=ax)
        else:
            sns.histplot(data=data, x=feat, hue=target, stat=stat, bins=bins, ax=ax)
