import traceback
from itertools import cycle
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy
from loguru import logger

from .colors import plotly_colors


multiclass_perc_format = "{target}={lab}(%)"
multiclass_se_format = "{target}={lab}(se)"
binary_perc_format = "%{target}=1"
binary_se_format = "{target}(se)%"
regression_mean_format = "{target}(avg)"
regression_se_format = "{target}(se)"

predicted_proba_format = "{target}(avg)"
predicted_proba_se_format = "{target}(se)"

confidence = 0.95  # Change to your desired confidence level
z_value = scipy.stats.norm.ppf((1 + confidence) / 2.0)

alpha = 0.05


def t_value(df):
    return scipy.stats.t.ppf(1 - alpha / 2, df)


def get_equal_width_bins(data_feat: pd.Series, nbins=10) -> List:
    min_value = min(data_feat)
    max_value = max(data_feat)
    delta = max_value - min_value
    breaks = [min_value + 0.1 * i * delta for i in range(nbins + 1)]
    return breaks


def get_quantile_bins(data_feat: pd.Series, nbins=10) -> List:
    quantiles = np.linspace(start=0, stop=1, num=nbins)
    breaks = [data_feat.quantile(q) for q in quantiles]
    breaks[0] = breaks[0] * 0.99 if breaks[0] > 0 else breaks[0] * 1.01
    # print(data_feat)
    # print(breaks, quantiles)
    breaks[0] = -np.inf
    breaks[-1] = np.inf
    return breaks


def classification_plotly_plot_numerical_feature(data, feat, target, plot_df, error_bars=False):
    fig = go.Figure()

    target_rate_col = binary_perc_format.format(target=target[0] if isinstance(target, list) else target)
    target_se_col = binary_se_format.format(target=target[0] if isinstance(target, list) else target)

    fig.add_trace(
        go.Bar(
            x=plot_df.index,
            y=plot_df["counts"],
            marker_color="#1f77b4",
            text=plot_df["perc"] * 100,
            texttemplate="%{text:.2f}%",
            textposition="outside",
            name="Counts",
            hovertemplate="%{y:.2d}(%{text:.2f}%)",
        )
    )

    fig.add_trace(
        go.Bar(
            x=plot_df.index,
            y=plot_df["counts"] * plot_df[target_rate_col],
            marker_color="#9addbd",
            text=plot_df["counts"] * plot_df[target_rate_col],
            texttemplate="%{text:.1d}",
            textposition="inside",
            name=f"Count {target_rate_col.replace('%','#')}",
            hovertemplate="%{y:.2d})",
        )
    )

    # Target distribution plot across features
    if isinstance(target, str):
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df[target_rate_col],
                name=f"Perc {target_rate_col}",
                text=plot_df[target_rate_col].apply(str),
                error_y=dict(
                    type="data",  # value of error bar given in data coordinates,
                    array=round(plot_df[target_se_col], 4) * 1,
                    color="grey",
                )
                if error_bars
                else None,
                # customdata=[round(eda_data["pos_rate_v"],4)*1],
                marker_color="#ff7f0e",
                texttemplate="%{text:.1%}",
                textposition="bottom right",
                hovertemplate="%{y:.1%} (\u00b1%{error_y.array:.1%})" if error_bars else "%{y:.1%}",
                # mode='lines+text+markers',
                textfont=dict(color="#ff7f0e"),  # Text styling with red color
                yaxis="y2",
            ),
        )
    else:
        colors = cycle(plotly_colors)
        for target_y in target:
            if "predicted_proba" in target_y:
                target_rate_col = predicted_proba_format.format(target=target_y)
                target_se_col = predicted_proba_se_format.format(target=target_y)
            else:
                target_rate_col = binary_perc_format.format(target=target_y)
                target_se_col = binary_se_format.format(target=target_y)

            fig.add_trace(
                go.Scatter(
                    x=plot_df.index,
                    y=plot_df[target_rate_col],
                    name=f"{target_rate_col}",
                    text=plot_df[target_rate_col].apply(str),
                    error_y=dict(
                        type="data",  # value of error bar given in data coordinates,
                        array=round(plot_df[target_se_col], 4) * 1,
                        color="grey",
                    )
                    if error_bars
                    else None,
                    # customdata=[round(eda_data["pos_rate_v"],4)*1],
                    marker_color=next(colors),
                    texttemplate="%{text:.1%}",
                    textposition="bottom right",
                    hovertemplate="%{y:.1%} (\u00b1%{error_y.array:.1%})" if error_bars else "%{y:.1%}",
                    # mode='lines+text+markers',
                    textfont=dict(color="#ff7f0e"),  # Text styling with red color
                    yaxis="y2",
                ),
            )

    # Here we modify the tickangle of the xaxis, resulting in rotated labels.
    # fig.update_layout(barmode='group', xaxis_tickangle=-45)
    fig.update_layout(
        xaxis=dict(title=f"{feat}"),
        yaxis=dict(automargin=True),
        yaxis1=dict(
            title="Count",
            color="#1f77b4",
            range=[0, max(plot_df["counts"]) * 1.1],  # Extend the y-axis
        ),
        yaxis2=dict(
            title="Rate",
            color="#ff7f0e",
            overlaying="y",
            side="right",
        ),
    )

    # Update layout properties
    fig.update_layout(
        title=dict(
            text=f"<b>{feat.upper()}</b>",
            x=0.5,
            y=0.95,
            xanchor="center",
            yanchor="top",
            subtitle=dict(
                text="This is showing for some reason",
                font=dict(color="gray", size=13),
            ),
        ),
        # margin=dict(l=50, r=50, t=0, b=50),  # Adjust margins for better layout
        # margin=dict(
        #     pad=20,
        # ),
        legend={
            "title.text": "Legend",
            "font.variant": "small-caps",
            "x": 0.9,
            "y": 1.4,
        },
        hovermode="x unified",
        # height=800,
    )
    fig.update_yaxes(automargin=True)
    fig.update_xaxes(automargin=True)
    fig.show()
    return plot_df


def classification_multiclass_plotly_plot_numerical_feature(data, feat, target, plot_df, error_bars=False):
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=plot_df.index,
            y=plot_df["counts"],
            marker_color="#1f77b4",
            text=plot_df["perc"] * 100,
            texttemplate="%{text:.2f}%",
            textposition="inside",
            name="Counts",
            hovertemplate="%{y:.2d}(%{text:.2f}%)",
        )
    )

    unique_targets = sorted(data[target].unique())
    colors = cycle(plotly_colors)
    for lab in unique_targets:
        target_rate_col = multiclass_perc_format.format(target=target, lab=lab)
        target_se_col = multiclass_se_format.format(target=target, lab=lab)

        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df[target_rate_col],
                name=target_rate_col,
                text=plot_df[target_rate_col],
                error_y=dict(
                    type="data",  # value of error bar given in data coordinates,
                    array=round(plot_df[target_se_col], 4) * 1,
                    color="grey",
                )
                if error_bars
                else None,
                customdata=(plot_df[target_rate_col] * plot_df["counts"]).rename("cnt").to_frame(),
                yaxis="y2",
                marker_color=next(colors),
                texttemplate="%{text:.1%}%",
                # textposition="top center",
                hovertemplate="%{y:.1%} (\u00b1%{error_y.array:.1%})"
                if error_bars
                else "%{y:.1%} (%{customdata[0]:.1d})",
            ),
        )

    # Here we modify the tickangle of the xaxis, resulting in rotated labels.
    # fig.update_layout(barmode='group', xaxis_tickangle=-45)
    fig.update_layout(
        xaxis=dict(title=f"{feat}"),
        yaxis=dict(automargin=True),
        yaxis1=dict(
            title="Count",
            color="#1f77b4",
        ),
        yaxis2=dict(
            title="Percentage",
            color="#ff7f0e",
            overlaying="y",
            side="right",
        ),
    )

    # Update layout properties
    fig.update_layout(
        title=dict(
            text=f"<b>{feat.upper()}</b>",
            x=0.5,
            y=0.95,
            xanchor="center",
            yanchor="top",
            subtitle=dict(
                text="This is showing for some reason",
                font=dict(color="gray", size=13),
            ),
        ),
        # margin=dict(l=50, r=50, t=0, b=50),  # Adjust margins for better layout
        # margin=dict(
        #     pad=20,
        # ),
        legend={
            "title.text": "Legend",
            "font.variant": "small-caps",
            "x": 1.1,
            "y": 1.3,
        },
        hovermode="x unified",
        # height=800,
    )
    fig.update_yaxes(automargin=True)
    fig.update_xaxes(automargin=True)
    fig.show()
    return plot_df


def regression_plotly_plot_numerical_feature(data, feat, target, plot_df, error_bars=False):
    fig = go.Figure()

    target_mean_col = regression_mean_format.format(target=target)
    target_se_col = regression_se_format.format(target=target)

    fig.add_trace(
        go.Bar(
            x=plot_df.index,
            y=plot_df["counts"],
            marker_color="#1f77b4",
            text=plot_df["perc"],
            texttemplate="%{text:.1%}",
            textposition="outside",
            name="Counts",
            hovertemplate="%{y:.2d}",
        )
    )

    # fig.add_trace(
    #     go.Bar(
    #         x=plot_df.index,
    #         y=plot_df["counts"] * plot_df[target_mean_col],
    #         marker_color="#9addbd",
    #         text=plot_df["counts"] * plot_df[target_mean_col],
    #         texttemplate="%{text:.1d}",
    #         textposition="inside",
    #         name=f"Count {target_mean_col.replace('%','#')}",
    #         hovertemplate="%{y:.2d})",
    #     )
    # )

    # Target distribution plot across features
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df[target_mean_col],
            name=target_mean_col,
            text=plot_df[target_mean_col].apply(str),
            error_y=dict(
                type="data",  # value of error bar given in data coordinates,
                array=round(plot_df[target_se_col], 4) * 1,
                color="grey",
            )
            if error_bars
            else None,
            marker_color="#ff7f0e",
            texttemplate="%{text:.2f}",
            textposition="bottom right",
            hovertemplate="%{y:.2f} (\u00b1%{error_y.array:.2f})" if error_bars else "%{y:.2f}",
            # mode='lines+text+markers',
            textfont=dict(color="#ff7f0e"),  # Text styling with red color
            yaxis="y2",
        ),
    )

    # Here we modify the tickangle of the xaxis, resulting in rotated labels.
    # fig.update_layout(barmode='group', xaxis_tickangle=-45)
    fig.update_layout(
        xaxis=dict(title=f"{feat}"),
        yaxis=dict(automargin=True),
        yaxis1=dict(
            title="Count",
            color="#1f77b4",
            range=[0, max(plot_df["counts"]) * 1.1],  # Extend the y-axis
        ),
        yaxis2=dict(
            title="Mean",
            color="#ff7f0e",
            overlaying="y",
            side="right",
        ),
    )

    # Update layout properties
    fig.update_layout(
        title=dict(
            text=f"<b>{feat.upper()}</b>",
            x=0.5,
            y=0.95,
            xanchor="center",
            yanchor="top",
            subtitle=dict(
                text="This is showing for some reason",
                font=dict(color="gray", size=13),
            ),
        ),
        # margin=dict(l=50, r=50, t=0, b=50),  # Adjust margins for better layout
        # margin=dict(
        #     pad=20,
        # ),
        legend={
            "title.text": "Legend",
            "font.variant": "small-caps",
            "x": 1.0,
            "y": 1.3,
        },
        hovermode="x unified",
        # height=800,
    )
    fig.update_yaxes(automargin=True)
    fig.update_xaxes(automargin=True)
    fig.show()
    return plot_df


def plt_plot_numerical_feature_distribution(
    data,
    feat,
    target,
    nbins=100,
    equal_quantiles=True,
    show_percentages=True,
    remove_overlap=False,
):
    """dd = plot_numerical_feature_distribution(
        data, feat, target=response, nbins=10, equal_quantiles=False, remove_overlap=False
    )

        Args:
            data (_type_): _description_
            feat (_type_): _description_
            target (_type_): _description_
            nbins (int, optional): _description_. Defaults to 100.
            equal_quantiles (bool, optional): _description_. Defaults to True.
            show_percentages (bool, optional): _description_. Defaults to True.
            remove_overlap (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
    """

    fig_h, fig_w = 12, 18
    # fontsize = fig_w // 2
    label_fontsize = fig_w * 1.2
    title_fontsize = fig_w * 1.2
    suptitle_fontsize = fig_w * 1.5
    tick_fontsize = fig_w * 0.9
    ax1_color = "C1"
    ax2_color = "darkorange"

    texts = []
    if equal_quantiles:
        breaks = get_quantile_bins(data[feat], nbins=nbins)
    else:
        breaks = get_equal_width_bins(data[feat], nbins=nbins)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    data["binned"] = pd.cut(data[feat], bins=breaks, include_lowest=True, precision=1, duplicates="drop")
    data_feat_counts = data["binned"].value_counts(dropna=False)
    data_feat_counts = data_feat_counts.sort_index()
    data_feat_counts.plot(kind="bar", ax=ax, label="count", color="C1")

    if show_percentages:
        data_feat_perc = data["binned"].value_counts(dropna=False, normalize=True).sort_index()
        x_axis_labels = ax.get_xticks()
        for index, val in enumerate(data_feat_counts):
            text = plt.text(
                x_axis_labels[index],
                val + 1,
                f"{round(data_feat_perc.iloc[index]*100,2)}%",
                ha="center",
                fontsize=tick_fontsize * 0.9,
                color="grey",
                weight="bold",
            )

    ax2 = ax.twinx()

    if data[target].nunique() > 2:
        plotly_colors_iter = cycle(plotly_colors)
        for label in sorted(data[target].unique()):
            data_feat_pos_rate = (
                data[data[target] == label].groupby("binned", observed=False)[target].mean().sort_index()
            )
            data_feat_pos_rate.plot(ax=ax2, color=next(plotly_colors_iter), label=f"class={label}")
    else:
        data_feat_pos_rate = data.groupby("binned", observed=False)[target].mean().sort_index()
        data_feat_pos_rate.plot(ax=ax2, color=ax2_color, label=f"{target} (rate)", marker="X")

        if show_percentages:
            x_axis_labels = ax2.get_xticks()
            if remove_overlap:
                for index, val in enumerate(data_feat_pos_rate):
                    text = plt.text(
                        x_axis_labels[index],
                        1.05,
                        f"{round(data_feat_pos_rate.iloc[index]*100,2)}%",
                        ha="center",
                        fontsize=tick_fontsize * 0.9,
                        color="orange",
                        weight="bold",
                    )
            else:
                for index, val in enumerate(data_feat_pos_rate):
                    text = plt.text(
                        x_axis_labels[index],
                        val - 0.05,
                        f"{round(data_feat_pos_rate.iloc[index]*100,2)}%",
                        ha="center",
                        fontsize=tick_fontsize * 0.9,
                        color="orange",
                        weight="bold",
                    )
                    texts.append(text)

    # y_ticks = ax2.get_yticks()
    # y_ticks = np.append(y_ticks, y_ticks[-1] + (y_ticks[-1]-y_ticks[-2]))
    # adjust_text(texts,x=ax2.get_xticks(),
    #             y=y_ticks, xlim=ax2.get_xlim(), ylim=ax2.get_ylim(), only_move={'text': 'y'}, force_text=0.001, max_move=0.1)

    # Configure first x-axis
    ax.set_xlabel(
        f"Edges of the {feat} bins (right-inclusive)",
        fontsize=label_fontsize,
        labelpad=10,
    )
    ax.set_ylabel("Raw Counts", fontsize=label_fontsize, labelpad=10, color=ax1_color)
    ax.set_title("Statistics", fontsize=title_fontsize, y=1.05)
    plt.suptitle(f"Distribution of feature: `{feat}`", fontsize=suptitle_fontsize)
    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment="right",
        fontsize=tick_fontsize,
    )
    plt.setp(ax.get_yticklabels(), fontsize=tick_fontsize, color=ax1_color)

    # configure 2nd x-axis
    ax2.grid(False)
    ax2.set_ylabel("Positive Rate", fontsize=label_fontsize, labelpad=10, color=ax2_color)
    plt.setp(ax2.get_yticklabels(), fontsize=tick_fontsize, color=ax2_color)
    fig.legend(bbox_to_anchor=(1.0, 0.89), fontsize=tick_fontsize, loc=2, borderaxespad=0)
    plt.tight_layout()
    return data_feat_pos_rate


def regression_get_numerical_feature_stats_df(data, feat, target=None, nbins=10, equal_quantiles=True, breaks=None):
    if target is not None:
        feat_data = data[[feat, target]].copy()
    else:
        feat_data = data[[feat]].copy()

    if breaks is not None:
        breaks = breaks
    elif equal_quantiles:
        breaks = get_quantile_bins(data[feat], nbins=nbins)
    else:
        breaks = get_equal_width_bins(data[feat], nbins=nbins)

    # bin counts
    feat_data["bin"] = pd.cut(
        feat_data[feat],
        bins=breaks,
        include_lowest=True,
        precision=1,
        duplicates="drop",
    ).rename("bin")

    feat_data_counts = feat_data["bin"].value_counts(dropna=False).rename("counts")
    feat_data_perc = feat_data["bin"].value_counts(dropna=False, normalize=True).rename("perc")

    stats_dfs = [feat_data_counts, feat_data_perc]
    # target counts and stats
    if target is not None:
        feat_data_target_mean = (
            feat_data.groupby("bin", observed=False)[target]
            .mean()
            .rename(regression_mean_format.format(target=target))
        )
        feat_data_target_se = z_value * (
            feat_data.groupby("bin", observed=False)[target].std() / np.sqrt(feat_data_counts)
        ).rename(regression_se_format.format(target=target))

        stats_dfs.append(feat_data_target_mean)
        stats_dfs.append(feat_data_target_se)

    feat_data_grouped = pd.concat(stats_dfs, axis=1).sort_index()
    feat_data_grouped.index = map(str, feat_data_grouped.index)

    return feat_data_grouped


def regression_get_categorical_feature_stats_df(
    data,
    feat,
    target=None,
):
    if target is not None:
        feat_data = data[[feat, target]].copy()
    else:
        feat_data = data[[feat]].copy()

    feat_data_counts = feat_data[feat].value_counts(dropna=False).rename("counts")
    feat_data_perc = feat_data[feat].value_counts(dropna=False, normalize=True).rename("perc")

    stats_dfs = [feat_data_counts, feat_data_perc]
    # target counts and stats

    if target is not None:
        feat_data_target_mean = (
            feat_data.groupby(feat, observed=False)[target].mean().rename(regression_mean_format.format(target=target))
        )
        feat_data_target_se = z_value * (
            feat_data.groupby(feat, observed=False)[target].std()
            # ** 2
            / np.sqrt(feat_data_counts)
        ).rename(regression_se_format.format(target=target))

        stats_dfs.append(feat_data_target_mean)
        stats_dfs.append(feat_data_target_se)

    feat_data_grouped = pd.concat(stats_dfs, axis=1).sort_index()
    feat_data_grouped.index = map(str, feat_data_grouped.index)

    return feat_data_grouped


def classification_get_numerical_feature_stats_df(
    data, feat, target=None, nbins=10, equal_quantiles=True, breaks=None
):
    if target is not None:
        if isinstance(target, list):
            feat_data = data[[feat] + target].copy()
        else:
            feat_data = data[[feat, target]].copy()
    else:
        feat_data = data[[feat]].copy()

    if breaks is not None:
        breaks = breaks
    elif equal_quantiles:
        breaks = get_quantile_bins(data[feat], nbins=nbins)
    else:
        breaks = get_equal_width_bins(data[feat], nbins=nbins)

    # bin counts
    feat_data["bin"] = pd.cut(
        feat_data[feat],
        bins=breaks,
        include_lowest=True,
        precision=0,
        duplicates="drop",
    ).rename("bin")
    # print(feat_data)
    feat_data_counts = feat_data["bin"].value_counts(dropna=False).rename("counts")
    feat_data_perc = feat_data["bin"].value_counts(dropna=False, normalize=True).rename("perc")

    stats_dfs = [feat_data_counts, feat_data_perc]
    # target counts and stats
    if target is not None and not isinstance(target, list):
        if feat_data[target].nunique() > 2:
            unique_targets = sorted(feat_data[target].unique())
            for lab in unique_targets:
                feat_data[f"Actual={lab}"] = np.where(feat_data[target] == lab, 1, 0)
                feat_data_target_perc = (
                    feat_data.groupby("bin", observed=False)[f"Actual={lab}"]
                    .mean()
                    .rename(multiclass_perc_format.format(target=target, lab=lab))
                )

                feat_data_target_se = z_value * np.sqrt(
                    (feat_data_target_perc * (1 - feat_data_target_perc)) / feat_data_counts
                )
                # feat_data_target_se = (
                #     feat_data.groupby("bin", observed=False)[f"Actual={lab}"]
                #     .std()
                #     .rename(multiclass_se_format.format(target=target, lab=lab))
                #     ** 2
                # )

                feat_data_target_se = feat_data_target_se.rename(multiclass_se_format.format(target=target, lab=lab))

                stats_dfs.append(feat_data_target_perc)
                stats_dfs.append(feat_data_target_se)

        else:
            if isinstance(target, str):
                feat_data_target_perc = (
                    feat_data.groupby("bin", observed=False, dropna=False)[target]
                    .mean()
                    .rename(binary_perc_format.format(target=target))
                )

                ## Formulae for variance for numerical feature
                # feat_data_target_se = (
                #     feat_data.groupby("bin", observed=False)[target]
                #     .std()
                #     .rename(binary_se_format.format(target=target))
                #     # ** 2
                # )

                feat_data_target_se = z_value * np.sqrt(
                    (feat_data_target_perc * (1 - feat_data_target_perc)) / feat_data_counts
                )

                ## Formulae for bernoulli distribution standard deviation
                # feat_data_target_se = np.sqrt(
                #     feat_data_target_perc * (1 - feat_data_target_perc)
                # )

                feat_data_target_se = feat_data_target_se.rename(binary_se_format.format(target=target))
                print(feat_data_target_se)
                stats_dfs.append(feat_data_target_perc)
                stats_dfs.append(feat_data_target_se)

    elif target is not None and isinstance(target, list):
        for target_y in target:
            if "predicted_proba" in target_y:
                target_rate_col = predicted_proba_format.format(target=target_y)
                target_se_col = predicted_proba_se_format.format(target=target_y)
            else:
                target_rate_col = binary_perc_format.format(target=target_y)
                target_se_col = binary_se_format.format(target=target_y)

            feat_data_target_perc = feat_data.groupby("bin", observed=False)[target_y].mean().rename(target_rate_col)

            feat_data_target_se = z_value * np.sqrt(
                (feat_data_target_perc * (1 - feat_data_target_perc)) / feat_data_counts
            ).rename(target_se_col)
            stats_dfs.append(feat_data_target_perc)
            stats_dfs.append(feat_data_target_se)

    feat_data_grouped = pd.concat(stats_dfs, axis=1).sort_index()
    feat_data_grouped.index = map(str, feat_data_grouped.index)

    return feat_data_grouped


def classification_get_categorical_feature_stats_df(
    data, feat, target=None, nbins=10, equal_quantiles=True, breaks=None
):
    if target is not None:
        feat_data = data[[feat, target]].copy()
    else:
        feat_data = data[[feat]].copy()

    feat_data_counts = feat_data[feat].value_counts(dropna=False).rename("counts")
    feat_data_perc = feat_data[feat].value_counts(dropna=False, normalize=True).rename("perc")

    stats_dfs = [feat_data_counts, feat_data_perc]
    # target counts and stats
    if target is not None:
        if feat_data[target].nunique() > 2:
            unique_targets = sorted(feat_data[target].unique())
            for lab in unique_targets:
                feat_data[f"Actual={lab}"] = np.where(feat_data[target] == lab, 1, 0)
                feat_data_target_perc = (
                    feat_data.groupby(feat, observed=False)[f"Actual={lab}"]
                    .mean()
                    .rename(multiclass_perc_format.format(target=target, lab=lab))
                )
                # feat_data_target_se= (
                #     feat_data.groupby("bin", observed=False)[f"Actual={lab}"]
                #     .std()
                #     .rename(multiclass_se_format.format(lab=lab))
                #     # ** 2
                # )

                feat_data_target_se = z_value * np.sqrt(
                    (feat_data_target_perc * (1 - feat_data_target_perc)) / feat_data_counts
                )

                feat_data_target_se = feat_data_target_se.rename(multiclass_se_format.format(target=target, lab=lab))

                stats_dfs.append(feat_data_target_perc)
                stats_dfs.append(feat_data_target_se)

        else:
            feat_data_target_perc = (
                feat_data.groupby(feat, observed=False)[target].mean().rename(binary_perc_format.format(target=target))
            )
            # feat_data_target_se = (
            #     feat_data.groupby(feat, observed=False)[target]
            #     .std()
            #     .rename(binary_se_format.format(target=target))
            #     # ** 2
            # )

            feat_data_target_se = z_value * np.sqrt(
                (feat_data_target_perc * (1 - feat_data_target_perc)) / feat_data_counts
            )
            feat_data_target_se = feat_data_target_se.rename(binary_se_format.format(target=target))

            stats_dfs.append(feat_data_target_perc)
            stats_dfs.append(feat_data_target_se)

    feat_data_grouped = pd.concat(stats_dfs, axis=1).sort_index()
    feat_data_grouped.index = map(str, feat_data_grouped.index)

    return feat_data_grouped


def check(**kwargs):
    feat = kwargs.get("feat", None)
    target = kwargs.get("target", None)
    if all(i is not None for i in [feat, target]):
        assert feat != target, f"feat `{feat}` is the same as target `{target}`"

    data = kwargs.get("data", None)
    assert data is not None, f"parameter `data` must be supplied as parameters. got data `{data}`"
    assert isinstance(data, pd.DataFrame), f"parameter `data` must be of type `{pd.DataFrame}`, got `{type(data)}`"

    nbins = kwargs.get("nbins", None)
    assert isinstance(nbins, int), f"parameter `nbins` must be of type `{int}`, got `{type(nbins)}`"


def plot_numerical_feature_distribution(
    data,
    feat,
    target,
    extra_targets: Optional[List] = None,
    nbins=10,
    equal_quantiles=True,
    error_bars=False,
    mode="classification",
    breaks=None,
):
    """_summary_

    plot_numerical_feature_distribution(data, feat=feat, target=response, equal_quantiles=True)


    Args:
        data (_type_): _description_
        feat (_type_): _description_
        target (_type_): _description_
        nbins (int, optional): _description_. Defaults to 10.
        equal_quantiles (bool, optional): _description_. Defaults to True.
    """

    if mode == "classification":
        # if there are special arugments
        if data[target].nunique() == 2 and extra_targets is not None and len(extra_targets) > 0:
            feat_data_plot_df = classification_get_numerical_feature_stats_df(
                data,
                feat,
                target=[target] + extra_targets,
                nbins=nbins,
                equal_quantiles=equal_quantiles,
                breaks=breaks,
            )
            classification_plotly_plot_numerical_feature(
                data, feat, [target] + extra_targets, feat_data_plot_df, error_bars=error_bars
            )
        else:
            feat_data_plot_df = classification_get_numerical_feature_stats_df(
                data, feat, target=target, nbins=nbins, equal_quantiles=equal_quantiles, breaks=breaks
            )
            if data[target].nunique() == 2:
                classification_plotly_plot_numerical_feature(
                    data, feat, target, feat_data_plot_df, error_bars=error_bars
                )

            else:
                classification_multiclass_plotly_plot_numerical_feature(
                    data, feat, target, feat_data_plot_df, error_bars=error_bars
                )

    elif mode == "regression":
        feat_data_plot_df = regression_get_numerical_feature_stats_df(
            data, feat, target=target, nbins=nbins, equal_quantiles=equal_quantiles, breaks=breaks
        )
        regression_plotly_plot_numerical_feature(data, feat, target, feat_data_plot_df, error_bars=error_bars)
    return feat_data_plot_df


def plot_categorical_feature_distribution(
    data,
    feat,
    target,
    nbins=10,
    equal_quantiles=True,
    error_bars=False,
    mode="classification",
    breaks=None,
):
    """_summary_

    plot_numerical_feature_distribution(data, feat=feat, target=response, equal_quantiles=True)


    Args:
        data (_type_): _description_
        feat (_type_): _description_
        target (_type_): _description_
        nbins (int, optional): _description_. Defaults to 10.
        equal_quantiles (bool, optional): _description_. Defaults to True.
    """
    check(
        data=data,
        feat=feat,
        target=target,
        nbins=nbins,
        equal_quantiles=equal_quantiles,
        error_bars=error_bars,
        mode=mode,
    )
    try:
        if mode == "classification":
            feat_data_plot_df = classification_get_categorical_feature_stats_df(
                data,
                feat,
                target=target,
                nbins=nbins,
                equal_quantiles=equal_quantiles,
            )

            # binary classification
            if data[target].nunique() == 2:
                logger.info("binary classification detected.")
                classification_plotly_plot_numerical_feature(
                    data, feat, target, feat_data_plot_df, error_bars=error_bars
                )
            # multiclassification
            else:
                logger.info("multi classification detected.")
                classification_multiclass_plotly_plot_numerical_feature(
                    data, feat, target, feat_data_plot_df, error_bars=error_bars
                )
        elif mode == "regression":
            feat_data_plot_df = regression_get_categorical_feature_stats_df(
                data,
                feat,
                target,
            )
            regression_plotly_plot_numerical_feature(data, feat, target, feat_data_plot_df, error_bars=error_bars)
    except Exception as e:
        logger.info(e)
        traceback.print_exc()
        return feat_data_plot_df

    return feat_data_plot_df
