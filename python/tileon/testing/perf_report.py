from __future__ import annotations

import os
from typing import List, Any, Dict, Callable

DEFAULT_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]
DEFAULT_LINESTYLES = ['-', '--', '-.', ':']
DEFAULT_FIGSIZE = (10, 6)
DEFAULT_FONT_NAME = 'Kristen ITC'
DEFAULT_DPI = 300


def _add_watermark(ax, logo_path=None, text='Tileon', alpha=0.15):
    """Add watermark to plot.

    Tries to use logo first, falls back to text if logo is not available.

    Args:
        ax: Matplotlib axes object to add watermark to.
        logo_path: Path to logo image file. If None or not found, uses text instead.
        text: Text to use as watermark if logo is not available. Defaults to 'Tileon'.
        alpha: Transparency of the watermark. Defaults to 0.15.
    """
    try:
        x0, y0, width, height = ax.get_position().bounds
        x = x0 + width / 2
        y = y0 + height / 2
        if logo_path and os.path.exists(logo_path):
            from PIL import Image
            img = Image.open(logo_path)
            fig = ax.figure
            fig.figimage(
                img,
                fig.bbox.xmin + x * fig.bbox.width - img.width / 2,
                fig.bbox.ymin + y * fig.bbox.height - img.height / 2,
                alpha=alpha,
                zorder=0
            )
        else:
            ax.text(
                x, y, text,
                fontsize=100,
                fontweight='bold',
                fontname=DEFAULT_FONT_NAME,
                color='gray',
                alpha=alpha,
                ha='center',
                va='center',
                transform=ax.figure.transFigure,
                zorder=0
            )
    except Exception:
        pass


class Benchmark:
    """A configuration class used by the :code:`perf_report` function
    to generate line plots with a concise API.
    """

    def __init__(
        self,
        x_names: List[str],
        x_vals: List[Any],
        line_arg: str,
        line_vals: List[Any],
        line_names: List[str],
        plot_name: str,
        args: Dict[str, Any],
        xlabel: str = '',
        ylabel: str = '',
        x_log: bool = False,
        y_log: bool = False,
        styles: List[tuple[str, str]] | None = None,
    ):
        """Initialize a Benchmark configuration.

        x_vals can be a list of scalars or a list of tuples/lists. If x_vals is a list
        of scalars and there are multiple x_names, all arguments will have the same value.
        If x_vals is a list of tuples/lists, each element should have the same length as
        x_names.

        Args:
            x_names: Name of the arguments that should appear on the x axis of the plot.
            x_vals: List of values to use for the arguments in :code:`x_names`.
            line_arg: Argument name for which different values correspond to different lines in the plot.
            line_vals: List of values to use for the arguments in :code:`line_arg`.
            line_names: Label names for the different lines.
            plot_name: Name of the plot.
            args: Dictionary of keyword arguments to remain fixed throughout the benchmark.
            xlabel: Label for the x axis of the plot. Defaults to ''.
            ylabel: Label for the y axis of the plot. Defaults to ''.
            x_log: Whether the x axis should be log scale. Defaults to False.
            y_log: Whether the y axis should be log scale. Defaults to False.
            styles: A list of tuples, where each tuple contains two elements: a color and a linestyle.
        """
        self.x_names = x_names
        self.x_vals = x_vals
        self.x_log = x_log
        self.line_arg = line_arg
        self.line_vals = line_vals
        self.line_names = line_names
        self.y_log = y_log
        self.styles = styles
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.plot_name = plot_name
        self.args = args


class Mark:
    """A class used by the :code:`perf_report` function
    to generate line plots with a concise API.
    """

    def __init__(self, fn: Callable, benchmarks: List[Benchmark]):
        """Initialize a Mark instance.

        Args:
            fn: Function to benchmark.
            benchmarks: List of Benchmark configurations.
        """
        self.fn = fn
        self.benchmarks = benchmarks

    def _run(self,
        bench: Benchmark,
        save_path: str,
        show_plots: bool,
        print_data: bool,
        diff_col: bool = False,
        save_precision: int = 6,
        **kwrags
    ) -> "pd.DataFrame":
        """Run a single benchmark.

        Args:
            bench: Benchmark configuration.
            save_path: Directory path to save the plot and data.
            show_plots: Whether to display the plots.
            print_data: Whether to print the data table.
            diff_col: Whether to add a difference column for two-column data. Defaults to False.
            save_precision: Number of decimal places for saved CSV. Defaults to 6.
            **kwrags: Additional keyword arguments to pass to the benchmark function.

        Returns:
            DataFrame containing the benchmark results.
        """
        import matplotlib.pyplot as plt
        import pandas as pd

        y_mean_labels = [f'{y} ({bench.ylabel})' for y in bench.line_names]
        y_min_labels = [f'{y}-min ({bench.ylabel})' for y in bench.line_names]
        y_max_labels = [f'{y}-max ({bench.ylabel})' for y in bench.line_names]
        x_names = list(bench.x_names)
        df = pd.DataFrame(columns=x_names + y_mean_labels + y_min_labels + y_max_labels)

        for x in bench.x_vals:
            # x can be a single value or a sequence of values.
            if not isinstance(x, (list, tuple)):
                x = [x for _ in x_names]
            if len(x) != len(x_names):
                raise ValueError(f"Expected {len(x_names)} values, got {x}")
            x_args = dict(zip(x_names, x))

            row_mean, row_min, row_max = [], [], []
            for y in bench.line_vals:
                ret = self.fn(**x_args, **{bench.line_arg: y}, **bench.args, **kwrags)
                try:
                    y_mean, y_min, y_max = ret
                except TypeError:
                    y_mean, y_min, y_max = ret, None, None
                row_mean += [y_mean]
                row_min += [y_min]
                row_max += [y_max]
            df.loc[len(df)] = list(x) + row_mean + row_min + row_max

        if bench.plot_name:
            plt.figure(figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI)
            ax = plt.gca()
            ax.grid(True, linestyle='--', alpha=0.6, zorder=0)
            ax.set_axisbelow(True)

            logo_path = None
            possible_paths = [
                os.path.join(os.path.dirname(__file__), '..', '_static', 'tileon-logo.png'),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    logo_path = path
                    break

            _add_watermark(ax, logo_path=logo_path)

            # Plot first x value on x axis if there are multiple.
            first_x = x_names[0]
            for i, (mean_label, min_label, max_label) in enumerate(
                    zip(y_mean_labels, y_min_labels, y_max_labels)
                ):
                y_min, y_max = df[min_label], df[max_label]
                if bench.styles:
                    col = bench.styles[i][0]
                    sty = bench.styles[i][1]
                else:
                    col = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
                    sty = DEFAULT_LINESTYLES[i % len(DEFAULT_LINESTYLES)]

                ax.plot(
                    df[first_x],
                    df[mean_label],
                    label=mean_label,
                    color=col,
                    linestyle=sty,
                    linewidth=2,
                    marker='o',
                    markersize=4,
                    zorder=3
                )
                if not y_min.isnull().all() and not y_max.isnull().all():
                    y_min = y_min.astype(float)
                    y_max = y_max.astype(float)
                    ax.fill_between(df[first_x], y_min, y_max, alpha=0.2, color=col, zorder=1)

            ax.legend(loc='best', framealpha=0.9, fancybox=True, shadow=True, fontsize=10)
            ax.set_xlabel(bench.xlabel or first_x, fontsize=11, fontweight='medium')
            ax.set_ylabel(bench.ylabel, fontsize=11, fontweight='medium')
            ax.set_title(bench.plot_name, fontsize=13, fontweight='bold', pad=15)
            ax.set_xscale("log" if bench.x_log else "linear")
            ax.set_yscale("log" if bench.y_log else "linear")
            plt.tight_layout()

            if show_plots:
                plt.show()
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(
                    os.path.join(save_path, f"{bench.plot_name}.png"),
                    bbox_inches='tight',
                    dpi=DEFAULT_DPI
                )
                plt.close()

        df = df[x_names + y_mean_labels]
        if diff_col and df.shape[1] == 2:
            col0, col1 = df.columns.tolist()
            df['Diff'] = df[col1] - df[col0]

        if print_data:
            print(bench.plot_name + ':')
            print(df.to_string())
        if save_path:
            df.to_csv(
                os.path.join(save_path, f"{bench.plot_name}.csv"),
                float_format=f"%.{save_precision}f",
                index=False
            )
        return df

    def run(
        self,
        show_plots: bool = False,
        print_data: bool = False,
        save_path: str = '',
        return_df: bool = False,
        **kwargs
    ) -> "pd.DataFrame" | None:
        """Run the benchmarks.

        Args:
            show_plots: Whether to display the plots. Defaults to False.
            print_data: Whether to print the data tables. Defaults to False.
            save_path: Directory path to save the plots and data. Defaults to ''.
            return_df: Whether to return the DataFrame(s). Defaults to False.
            **kwargs: Additional keyword arguments to pass to the benchmark function.

        Returns:
            If return_df is True:
                - Single DataFrame if only one benchmark
                - List of DataFrames if multiple benchmarks
            None otherwise.
        """
        has_single_bench = isinstance(self.benchmarks, Benchmark)
        benchmarks = [self.benchmarks] if has_single_bench else self.benchmarks
        result_dfs = []
        try:
            for bench in benchmarks:
                result_dfs.append(self._run(bench, save_path, show_plots, print_data, **kwargs))
        finally:
            if save_path:
                with open(os.path.join(save_path, "results.html"), "w") as html:
                    html.write("<html><body>\n")
                    for bench in benchmarks[:len(result_dfs)]:
                        html.write(f"<image src=\"{bench.plot_name}.png\"/>\n")
                    html.write("</body></html>\n")
        if return_df:
            if has_single_bench:
                return result_dfs[0]
            else:
                return result_dfs
        return None


def perf_report(benchmarks: List[Benchmark]):
    """Mark a function for benchmarking.

    The benchmark can then be executed by using the :code:`.run` method on the return value.

    Args:
        benchmarks: Benchmarking configurations. Can be a single Benchmark or a list of Benchmarks.

    Returns:
        Decorated function wrapped in a Mark object.
    """
    wrapper = lambda fn: Mark(fn, benchmarks)
    return wrapper
