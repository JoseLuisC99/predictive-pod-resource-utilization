import os
import argparse
import json
import requests
import pandas as pd
import numpy as np
from typing import Dict
from tqdm.contrib.itertools import product
from tqdm import tqdm

PROMETHEUS_ENDPOINT = 'http://localhost:9090/api/v1/query_range'
METRIC_QUERY = 'sum by (pod) (rate({}{{pod="{}"}}[{}]))'
GRAPH_QUERY = 'sum by (source_workload, destination_workload) (rate(istio_requests_total[{}]))'


def get_prometheus_data(query: str, start: int, end: int, resolution: int) -> Dict:
    response = requests.get(PROMETHEUS_ENDPOINT, {
        "query": query, "start": start, "end": end, "step": resolution
    })
    response.raise_for_status()

    response = json.loads(response.text)
    return response["data"]["result"]


def get_prometheus_metrics(metric: str, pod: str, step: str, start: int, end: int, resolution: int) -> Dict:
    query = METRIC_QUERY.format(metric, pod, step)
    return get_prometheus_data(query, start, end, resolution)


def get_prometheus_graph(step: str, start: int, end: int, resolution: int) -> Dict:
    query = GRAPH_QUERY.format(step)
    return get_prometheus_data(query, start, end, resolution)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prometheus metrics extractor')
    parser.add_argument("--start", required=True, type=int,
                        help="Start timestamp, inclusive")
    parser.add_argument("--end", required=True, type=int,
                        help=" End timestamp, inclusive.")
    parser.add_argument("--resol", default=900, type=int,
                        help="Query resolution step width (in seconds).")
    parser.add_argument("--step", default="2h", type=str,
                        help="Query interval length for rate.")
    parser.add_argument("--output", required=True, type=str,
                        help="Output pathname.")
    parser.add_argument("--window", default=12, type=int,
                        help="Number of previous queries to consider to make a prediction (The time duration is "
                             "window * resol seconds)")
    parser.add_argument("--horizont", default=1, type=int,
                        help="Next value to predict (horizont * resol seconds after the resolution windows)")
    args = parser.parse_args()

    metric_info = [
        ("container_cpu_usage_seconds_total", "cpu"),
        ("container_memory_usage_bytes", "mem")
    ]
    pod_info = [
        ("adservice-b64d6db99-6qlr7", "adservice"),
        ("cartservice-7bdffbf7-nzwgf", "cartservice"),
        ("checkoutservice-7988bbf57d-ql2vx", "checkoutservice"),
        ("currencyservice-6f58f94d86-mf5pn", "currencyservice"),
        ("emailservice-965c88745-2zv7z", "emailservice"),
        ("frontend-7cc759d45-7l2ff", "frontend"),
        ("paymentservice-6cff547576-h4qsq", "paymentservice"),
        ("productcatalogservice-6b7556db5-t2sfh", "productcatalogservice"),
        ("recommendationservice-85555c666b-hhp27", "recommendationservice"),
        ("redis-cart-5477c6b974-vsvd5", "redis-cart"),
        ("shippingservice-68fc55fb6d-plfqx", "shippingservice"),
    ]

    print("Loading Prometheus pod metrics")
    node_df = []
    for ((metric_id, metric_name), (pod_id, pod_name)) in tqdm(product(metric_info, pod_info)):
        result = get_prometheus_metrics(metric_id, pod_id, args.step, args.start, args.end, args.resol)
        values = result[0]["values"]
        df = pd.DataFrame(values, columns=["timestamp", f"{pod_name}-{metric_name}"]).set_index("timestamp")
        node_df.append(df)
    node_df = pd.concat(node_df, axis=1, join="inner").apply(pd.to_numeric)
    node_df.to_csv(os.path.join(args.output, "pod_metrics.csv"))

    print("Loading Prometheus pod requests")
    graph_query = get_prometheus_graph(args.step, args.start, args.end, args.resol)
    graph_df = []
    for result in tqdm(graph_query):
        source = result["metric"]["source_workload"]
        dest = result["metric"]["destination_workload"]

        if source == "unknown" or dest == "unknown" or source == "loadgenerator" or source == "loadgenerator":
            continue
        for [timestamp, value] in result["values"]:
            graph_df.append([timestamp, source, dest, float(value)])
    graph_df = pd.DataFrame(graph_df, columns=["timestamp", "from", "to", "value"])
    graph_df = graph_df[graph_df["timestamp"].isin(node_df.index)].set_index("timestamp")
    graph_df.to_csv(os.path.join(args.output, "pod_requests.csv"))

    node_df = node_df.sort_values("timestamp")
    graph_df = graph_df.sort_values("timestamp")

    # Node info extractor section
    print("Generating node information")
    X_node = []
    y_node = []

    for i in tqdm(range(args.window, len(node_df) - args.horizont + 1)):
        node_window = node_df.iloc[i - args.window: i]
        node_horizont = node_df.iloc[i + args.horizont - 1]

        x, y = [], []
        for (_, pod_name) in pod_info:
            x.append(node_window[[f"{pod_name}-cpu", f"{pod_name}-mem"]].to_numpy())
            y.append(node_horizont[[f"{pod_name}-cpu", f"{pod_name}-mem"]].to_numpy())
        X_node.append(np.array(x).swapaxes(0, 1))
        y_node.append(np.array(y))
    X_node = np.array(X_node)
    y_node = np.array(y_node)
    X_node[:, :, :, 1] = X_node[:, :, :, 1] / 1e6
    y_node[:, :, 1] = y_node[:, :, 1] / 1e6
    np.savez(os.path.join(args.output, "node_features.npz"), X=X_node, y=y_node)
    print("Node dataset shape: X =", X_node.shape, ", y =", y_node.shape)

    # Edge info extractor section
    print("Generating edge information")
    edge_timed = graph_df.groupby("timestamp")
    A_graph = []
    graph = []

    for k in tqdm(range(len(node_df) - args.horizont)):
        node = node_df.iloc[k]

        edges_info = edge_timed.get_group(node.name)
        A = np.zeros((len(pod_info), len(pod_info)))
        for i in range(len(pod_info)):
            for j in range(len(pod_info)):
                from_pod = pod_info[i][1]
                to_pod = pod_info[j][1]

                query = edges_info.query("`from` == @from_pod and `to` == @to_pod")
                if len(query) > 0:
                    A[i, j] = query["value"].item()
        graph.append(A)

        if len(graph) == args.window:
            A_graph.append(graph)
            graph = graph[1:]
    A_graph = np.array(A_graph)
    np.savez(os.path.join(args.output, "edge_features.npz"), A=A_graph)
    print("Edge dataset shape:", A_graph.shape)
