
import json
import re

import numpy as np
import cv2


def calculate_blank_ratio(image_path, threshold=245):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error: Image not found or invalid!")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    white_pixels = cv2.countNonZero(mask)

    total_pixels = image.shape[0] * image.shape[1]
    blank_ratio = white_pixels / total_pixels

    return blank_ratio

def do_lines_intersect(x1, y1, x2, y2):
    def is_on_segment(px, py, ax, ay, bx, by):
        return min(ax, bx) <= px <= max(ax, bx) and min(ay, by) <= py <= max(ay, by)
    
    def orientation(ax, ay, bx, by, cx, cy):
        val = (by - ay) * (cx - bx) - (bx - ax) * (cy - by)
        if val == 0:
            return 0  # collinear
        elif val > 0:
            return 1  # clockwise
        else:
            return 2  # counterclockwise
    
    def check_intersect(ax, ay, bx, by, cx, cy, dx, dy):
        o1 = orientation(ax, ay, bx, by, cx, cy)
        o2 = orientation(ax, ay, bx, by, dx, dy)
        o3 = orientation(cx, cy, dx, dy, ax, ay)
        o4 = orientation(cx, cy, dx, dy, bx, by)
        
        if o1 != o2 and o3 != o4:
            return True  # Proper intersection
        
        # Check collinear cases
        if o1 == 0 and is_on_segment(cx, cy, ax, ay, bx, by):
            return True
        if o2 == 0 and is_on_segment(dx, dy, ax, ay, bx, by):
            return True
        if o3 == 0 and is_on_segment(ax, ay, cx, cy, dx, dy):
            return True
        if o4 == 0 and is_on_segment(bx, by, cx, cy, dx, dy):
            return True
        
        return False
    
    n1 = len(x1)
    n2 = len(x2)
    
    for k in range(n1 - 1):
        for l in range(n2 - 1):
            if check_intersect(x1[k], y1[k], x1[k + 1], y1[k + 1], x2[l], y2[l], x2[l + 1], y2[l + 1]):
                return True
    
    return False

def process_axes(fig, axes, row, col):
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]]).reshape(1, 1)
        row, col = 1, 1
    elif isinstance(axes, np.ndarray) and axes.ndim == 1:
        axes = axes.reshape(row, col)
    
    for ax in axes.flatten():
        ax.figure.canvas.draw()  # Ensure the rendering is done

    # shared information
    # title_shared, xlabel_shared, ylabel_shared, legend_shared = "", "", "", []
    # if row > 1 or col > 1:
    title_shared = fig._suptitle.get_text() if fig._suptitle is not None else ""
    xlabel_shared = fig._supxlabel.get_text() if fig._supxlabel is not None else ""
    ylabel_shared = fig._supylabel.get_text() if fig._supylabel is not None else ""
    legend_shared = [le.texts[0].get_text() if len(le.texts) > 0 else "" for le in fig.legends]

    if isinstance(legend_shared, list) and len(legend_shared) > 0 and len(set(legend_shared)) < len(legend_shared):
        unique_legend_texts = list(dict.fromkeys(legend_shared))  # remove duplicates while keeping order
        
        existing_legend = fig.legends[0]  # Get the first legend; assuming there's only one
        
        # Get current handles and labels
        handles, labels = existing_legend.legend_handles, existing_legend.texts
        labels_texts = [text.get_text() for text in labels]
        
        # Filter handles and labels based on unique labels
        unique_handles_labels = [(h, l) for h, l in zip(handles, labels_texts) if l in unique_legend_texts]

        if len(unique_handles_labels) == 0:
            unique_handles = []
        else:
            # Unzip into separate handle and label lists
            unique_handles, _ = zip(*unique_handles_labels)
        
        # Remove the existing legend
        existing_legend.remove()

        # Create a new legend with unique handles and labels
        fig.legend(unique_handles, unique_legend_texts)
        legend_shared = unique_legend_texts

    output = {}
    subplot_loc = "the current plot"
    for i in range(row):
        for j in range(col):
            ax = axes[i, j]
            if row > 1 or col > 1:
                subplot_loc = "the subplot at row " + str(i+1) + " and column " + str(j+1)
            x_tick_labels = [label.get_text() for label in ax.get_xticklabels()]
            y_tick_labels = [label.get_text() for label in ax.get_yticklabels()]
            try:
                z_tick_labels = [label.get_text() for label in ax.get_zticklabels()]
            except AttributeError:
                z_tick_labels = []
            
            xlabel = ax.get_xlabel()
            ylabel = ax.get_ylabel()
            title = ax.get_title()
            legend = ax.get_legend()
            legend_texts = [text.get_text() for text in legend.get_texts()] if legend is not None else "Not Applicable"

            # sharing legends
            if len(legend_shared) > 0 and str(legend_shared) == str(legend_texts):
                # remove the legend if it is shared by all subplots
                if isinstance(legend_texts, list):
                    for text in legend_texts:
                        if text in legend_shared:
                            ax.legend().remove()
                            break
            elif len(legend_shared) > 0 and fig.legends:
                # fig.legend().remove()
                fig.legends[0].remove()

            # detect if there are duplicate legend texts
            if isinstance(legend_texts, list) and len(set(legend_texts)) < len(legend_texts):
                # Use dictionary to remove duplicates while keeping order
                unique_legend_texts = list(dict.fromkeys(legend_texts))

                # Update legend to include only unique labels
                # Get handles and associated labels
                handles, labels = ax.get_legend_handles_labels()
                # Filter handles based on unique labels
                unique_handles_labels = [(h, l) for h, l in zip(handles, labels) if l in unique_legend_texts]

                if len(unique_handles_labels) == 0:
                    unique_handles = []
                else:
                    # Unzip into separate handle and label lists
                    unique_handles, _ = zip(*unique_handles_labels)

                # Remove the existing legend
                ax.legend().remove()
                
                # Create a new legend with unique handles and labels
                ax.legend(unique_handles, unique_legend_texts)
                legend_texts = unique_legend_texts
            elif isinstance(legend_texts, list) and len(legend_texts) > 0:  # reset legend to avoid overlap or truncation
                handles, labels = ax.get_legend_handles_labels()
                if len(handles) > 0:
                    ax.legend().remove()
                    ax.legend(handles, legend_texts)
                else:
                    ax.legend(legend_texts)

            x_tick_labels = [str(label).replace('\u2212', '-') for label in x_tick_labels]
            y_tick_labels = [str(label).replace('\u2212', '-') for label in y_tick_labels]
            z_tick_labels = [str(label).replace('\u2212', '-') for label in z_tick_labels]
            
            x_tick_interval, y_tick_interval = "Not Applicable", "Not Applicable"
            if len(x_tick_labels) > 0 and bool(re.match(r'^-?\d+(\.\d+)?$', x_tick_labels[0])):
                try:
                    x_tick_labels = [round(float(label), 3) if '.' in str(label) and len(str(label).split('.')[1]) > 3 else float(label) for label in x_tick_labels]
                    x_tick_labels = [int(label) if isinstance(label, float) and label.is_integer() else label for label in x_tick_labels]
                    x_tick_interval = (x_tick_labels[1] - x_tick_labels[0])
                    x_tick_interval = round(float(x_tick_interval), 3) if '.' in str(x_tick_interval) and len(str(x_tick_interval).split('.')[1]) > 3 else x_tick_interval
                except Exception:
                    pass
            
            if len(y_tick_labels) > 0 and bool(re.match(r'^-?\d+(\.\d+)?$', y_tick_labels[0])):
                try:
                    y_tick_labels = [round(float(label), 3) if '.' in str(label) and len(str(label).split('.')[1]) > 3 else float(label) for label in y_tick_labels]
                    y_tick_labels = [int(label) if isinstance(label, float) and label.is_integer() else label for label in y_tick_labels]
                    y_tick_interval = (y_tick_labels[1] - y_tick_labels[0])
                    y_tick_interval = round(float(y_tick_interval), 3) if '.' in str(y_tick_interval) and len(str(y_tick_interval).split('.')[1]) > 3 else y_tick_interval
                except Exception:
                    pass
            
            if len(z_tick_labels) > 0 and bool(re.match(r'^-?\d+(\.\d+)?$', z_tick_labels[0])):
                try:
                    z_tick_labels = [round(float(label), 3) if '.' in str(label) and len(str(label).split('.')[1]) > 3 else float(label) for label in z_tick_labels]
                    z_tick_labels = [int(label) if isinstance(label, float) and label.is_integer() else label for label in z_tick_labels]
                    z_tick_interval = (z_tick_labels[1] - z_tick_labels[0])
                    z_tick_interval = round(float(z_tick_interval), 3) if '.' in str(z_tick_interval) and len(str(z_tick_interval).split('.')[1]) > 3 else z_tick_interval
                except Exception:
                    pass
            
            if len(ax.get_xticklabels()) < 20:
                ax.set_xticks([x for x in ax.get_xticks()])
                ax.set_xticklabels([label for label in x_tick_labels])
            else:
                interval = len(x_tick_labels) // 19
                indices = list(range(0, len(x_tick_labels), interval))
                # Ensure the last index is included
                if indices[-1] != len(x_tick_labels) - 1:
                    indices.append(len(x_tick_labels) - 1)

                ax.set_xticks([ax.get_xticks()[idx] for idx in indices])
                ax.set_xticklabels([x_tick_labels[idx] for idx in indices], rotation=90)

                original_size = fig.get_size_inches()
                fig.set_size_inches(original_size[0] * 1.2, original_size[1] * (1.5 + (0.6 // (row * col))))

            if len(ax.get_xticklabels()) > 5 and sum([len(label.get_text()) for label in ax.get_xticklabels()])/len(ax.get_xticklabels()) > 6:
                ax.set_xticks([x for x in ax.get_xticks()])
                ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
                original_size = fig.get_size_inches()
                fig.set_size_inches(original_size[0]*1.1, original_size[1]*(1.5+(0.6//(row*col))))

            if len(ax.get_yticklabels()) > 5 and sum([len(label.get_text()) for label in ax.get_yticklabels()])/len(ax.get_yticklabels()) > 6:
                original_size = fig.get_size_inches()
                fig.set_size_inches(original_size[0]*(1.2+(0.6//(row*col))), original_size[1])

            if len(ax.get_yticklabels()) < 30:
                ax.set_yticks([y for y in ax.get_yticks()])
                ax.set_yticklabels([label for label in y_tick_labels])
            else:
                interval = len(y_tick_labels) // 29  # 30 points mean 29 intervals
                indices = list(range(0, len(y_tick_labels), interval))
                # Ensure the last index is included
                if indices[-1] != len(y_tick_labels) - 1:
                    indices.append(len(y_tick_labels) - 1)
                ax.set_yticks([ax.get_yticks()[idx] for idx in indices])
                ax.set_yticklabels([y_tick_labels[idx] for idx in indices])

            if len(z_tick_labels) > 0:
                if len(z_tick_labels) < 30:
                    ax.set_zticks([z for z in ax.get_zticks()])
                    ax.set_zticklabels([label for label in z_tick_labels])
                else:
                    ax.set_zticks([ax.get_zticks()[idx] for idx in range(0, 30, 2)])
                    ax.set_zticklabels([z_tick_labels[idx] for idx in range(0, 30, 2)])

            if title.lower() in title_shared.lower():
                title = title_shared
                ax.set_title("")
            if xlabel.lower() in xlabel_shared.lower():
                xlabel = xlabel_shared
                ax.set_xlabel("")
            if ylabel.lower() in ylabel_shared.lower():
                ylabel = ylabel_shared
                ax.set_ylabel("")

            if title.strip() == "":
                if title_shared != "":
                    title = title_shared
                else:
                    title = "Not Applicable"
            
            if xlabel.strip() == "":
                if xlabel_shared != "":
                    xlabel = xlabel_shared
                else:
                    xlabel = "Not Applicable"
            if ylabel.strip() == "":
                if ylabel_shared != "":
                    ylabel = ylabel_shared
                else:
                    ylabel = "Not Applicable"

            # remove uninformative title and labels
            for name in ["title", "plot", "graph", "map", "chart", "histogram", "greek legends", "class", "cluster", "label", "none", "not", "n/a"]:
                if (title_shared != "" and name in title_shared.lower()) or title_shared.lower() in [xlabel.lower(), ylabel.lower()]:
                    title_shared = ""
                    title = ""
                    fig.suptitle("")
                
                if name in title.lower() or title.lower() in [xlabel.lower(), ylabel.lower()]:
                    ax.set_title("")
                    if title_shared != "":
                        title = title_shared
                    else:
                        title = "Not Applicable"       
                
                if ylabel == title or ylabel == title:
                    ax.set_title("")
                    title = "Not Applicable"
                    break

            for name in ["cluster", "class", "categor", "label", "axis", "value", "none", "subplot", "not", "n/a"]:
                if xlabel_shared != "" and name in xlabel_shared.lower():
                    xlabel_shared = ""
                    xlabel = "Not Applicable"
                    fig.supylabel("")
                
                if name in xlabel.lower():
                    ax.set_xlabel("")
                    if xlabel_shared != "":
                        xlabel = xlabel_shared
                    else:
                        xlabel = "Not Applicable"
            
            for name in ["label", "axis", "value", "not presented", "none", "subplot", "not", "n/a"]:
                if ylabel_shared != "" and name in ylabel_shared.lower():
                    ylabel_shared = ""
                    ylabel = "Not Applicable"
                    fig.supxlabel("")
                
                if name in ylabel.lower():
                    ax.set_ylabel("")
                    if ylabel_shared != "":
                        ylabel = ylabel_shared
                    else:
                        ylabel = "Not Applicable"
                    
            # this alogorithm is not perfect, but it should work for most cases
            num_lines, lines_trend, lines_intersec = "Not Applicable", "Not Applicable", "Not Applicable"
            if len(ax.lines) > 0:
                lines_intersec = "No."
                num_lines = len(ax.lines)

                x_data_all = []
                increase, decrease = 0, 0
                for line in ax.lines:
                    x_data = []
                    for x_d in line.get_xdata():
                        if isinstance(x_d, str):
                            x_data.append(len(x_data)+1)
                        else:
                            x_data.append(x_d)
                    x_data = np.array(x_data)
                    x_data_all.append(x_data)
                    y_data = line._y

                    if len(x_data) == 0 and len(y_data) == 0:
                        continue
                    
                    if (x_data[-1] - x_data[0]) * (y_data[-1] - y_data[0]) >= 0:
                        increase += 1
                    elif (x_data[-1] - x_data[0]) * (y_data[-1] - y_data[0]) < 0:
                        decrease += 1
                
                line_data = [(x_d, line.get_ydata()) for x_d, line in zip(x_data_all, ax.lines)]

                for k in range(num_lines):
                    for l in range(k+1, num_lines):
                        x1, y1 = line_data[k]
                        x2, y2 = line_data[l]
                        if do_lines_intersect(x1, y1, x2, y2):
                            lines_intersec = "Yes"
                            break
                    if lines_intersec == "Yes":
                        break
               
                if increase > 0 and increase >= decrease:
                    lines_trend = "Increases"  
                elif decrease > 0 and decrease > increase:
                    lines_trend = "Decreases"

            colorbar_exist, max_value_colorbar, min_value_colorbar, diff_colorbar = "Not Applicable", "Not Applicable", "Not Applicable", "Not Applicable"
            if len(ax.get_images()) > 0 and hasattr(ax.get_images()[0], 'colorbar'):
                colorbar = ax.get_images()[0].colorbar
                if colorbar is not None:
                    colorbar_exist = "Yes"
                    max_value_colorbar = colorbar.norm.vmax
                    min_value_colorbar = colorbar.norm.vmin
                    max_value_colorbar = round(float(max_value_colorbar), 2) if '.' in str(max_value_colorbar) and len(str(max_value_colorbar).split('.')[1]) > 2 else float(max_value_colorbar)
                    min_value_colorbar = round(float(min_value_colorbar), 2) if '.' in str(min_value_colorbar) and len(str(min_value_colorbar).split('.')[1]) > 2 else float(min_value_colorbar)
                    diff_colorbar = max_value_colorbar - min_value_colorbar
                    images = ax.get_images()

                    import matplotlib.colors as mcolors
                    image = images[0] 
                    image.set_norm(mcolors.Normalize(vmin=min_value_colorbar, vmax=max_value_colorbar))

                    colorbar.remove()
                    fig.colorbar(image, ax=ax)

            output[str(subplot_loc)] = {
                    'x_tick_labels': x_tick_labels,
                    'y_tick_labels': y_tick_labels,
                    'z_tick_labels': z_tick_labels,
                    'x_tick_difference': x_tick_interval,
                    'y_tick_difference': y_tick_interval,
                    'y_tick_highest': y_tick_labels[-1] if len(y_tick_labels) > 0 else "Not Applicable",
                    'y_tick_lowest': y_tick_labels[0] if len(y_tick_labels) > 0 else "Not Applicable",
                    'x_tick_rightmost': x_tick_labels[-1] if len(x_tick_labels) > 0 else "Not Applicable",
                    'x_tick_leftmost': x_tick_labels[0] if len(x_tick_labels) > 0 else "Not Applicable",
                    'num_ticks_xaxis': len(x_tick_labels),
                    'num_ticks_yaxis': len(y_tick_labels),
                    'num_ticks_zaxis': len(z_tick_labels),
                    'num_ticks_all_axes': len(x_tick_labels) + len(y_tick_labels) + len(z_tick_labels),
                    'label_of_xaxis': xlabel,
                    'label_of_yaxis': ylabel,
                    'title': title,
                    'labels_in_legend': legend_texts,
                    'num_lines': num_lines,
                    'trend_of_lines': lines_trend,
                    'intersection_of_lines': lines_intersec,
                    'colorbar': colorbar_exist,
                    'colorbar_min_value': min_value_colorbar,
                    'colorbar_max_value': max_value_colorbar,
                    'colorbar_diff': diff_colorbar
                }
    if row != 1 or col != 1:
        fig.set_tight_layout(True)
    
    return json.dumps(output, indent=2)