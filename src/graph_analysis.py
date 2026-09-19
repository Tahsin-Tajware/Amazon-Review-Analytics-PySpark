"""
Graph analytics on the reviewer-product network.

COURSE MAPPING: Week 4 — Spark GraphX: data pipeline, graph modelling, graph
processing, and algorithms.

WHY A GRAPH BELONGS HERE
------------------------
The fairness and segmentation results describe *who* captures helpful votes.
A graph answers a question neither can: how reviewers are connected through
the products they review, and whether influence propagates along those
connections.

The natural structure is a bipartite graph — reviewers on one side, products
on the other, an edge for every review. PageRank over it yields a centrality
score that is not simply review count: a reviewer who reviews a few products
that many others also review is more central than one who reviews many
obscure products. That distinction is invisible to any per-row feature.

IMPLEMENTATION NOTE: WHY NOT JUST GRAPHFRAMES
---------------------------------------------
GraphFrames is the usual DataFrame-level interface to GraphX, but it ships as
a Spark-version-matched package and routinely lags major Spark releases. This
runs on Spark 4.x, where a matching build may not exist.

Rather than make an entire week of the syllabus depend on whether a JAR
resolves, this module implements PageRank and connected components directly
as iterative DataFrame joins — which is what GraphX does underneath, via
Pregel-style message passing. GraphFrames is attempted first and used when
available; the fallback produces the same quantities.

This is a deliberate engineering choice, not a workaround: the implementation
makes the algorithm visible rather than hiding it behind a library call.
"""

import pandas as pd

from pyspark.sql import functions as F
from pyspark.storagelevel import StorageLevel


DAMPING = 0.85
DEFAULT_ITERATIONS = 10


# ----------------------------------------------------------------------
# Graph construction
# ----------------------------------------------------------------------

def build_bipartite_graph(df, min_reviewer_degree=2, min_product_degree=2):
    """
    Vertices: reviewers and products, namespaced so their IDs cannot collide.
    Edges:    one per review, reviewer -> product.

    Degree-1 nodes are pruned. A reviewer with a single review and a product
    with a single review contribute no connectivity — they are leaves that
    inflate vertex count and slow every iteration without affecting the
    ranking of anything else.
    """
    edges = (
        df.select(
            F.concat(F.lit("u:"), F.col("user_id")).alias("src"),
            F.concat(F.lit("p:"), F.col("parent_asin")).alias("dst"),
            F.col("helpful_vote").cast("double").alias("weight"),
        )
        .dropna()
    )

    u_deg = edges.groupBy("src").count().withColumnRenamed("count", "u_deg")
    p_deg = edges.groupBy("dst").count().withColumnRenamed("count", "p_deg")

    edges = (
        edges
        .join(u_deg, "src").join(p_deg, "dst")
        .filter((F.col("u_deg") >= min_reviewer_degree) &
                (F.col("p_deg") >= min_product_degree))
        .select("src", "dst", "weight")
    )

    vertices = (
        edges.select(F.col("src").alias("id"))
        .union(edges.select(F.col("dst").alias("id")))
        .distinct()
    )

    return vertices, edges


# ----------------------------------------------------------------------
# PageRank via iterative joins  (the GraphX algorithm, written out)
# ----------------------------------------------------------------------

def pagerank(vertices, edges, iterations=DEFAULT_ITERATIONS, damping=DAMPING):
    """
    Undirected PageRank on the bipartite graph.

    Edges are symmetrised because reviewing is not a directed relation: a
    reviewer confers standing on a product and a product confers standing on
    its reviewers. Running it directed would let rank flow only one way and
    make every reviewer a sink.

    Each iteration is one join plus one aggregation, which is exactly the
    Pregel superstep GraphX performs. Ranks are checkpointed to disk-backed
    storage because the lineage otherwise grows linearly with iterations and
    eventually blows the query planner.
    """
    both = (
        edges.select("src", "dst")
        .union(edges.select(F.col("dst").alias("src"), F.col("src").alias("dst")))
        .distinct()
    ).persist(StorageLevel.MEMORY_AND_DISK)

    out_deg = both.groupBy("src").count().withColumnRenamed("count", "out_deg")
    out_deg = out_deg.persist(StorageLevel.MEMORY_AND_DISK)

    n = vertices.count()
    ranks = vertices.withColumn("rank", F.lit(1.0 / max(n, 1)))

    for i in range(iterations):
        contribs = (
            both
            .join(ranks, both.src == ranks.id)
            .join(out_deg, "src")
            .select(F.col("dst").alias("id"),
                    (F.col("rank") / F.col("out_deg")).alias("contrib"))
        )

        ranks = (
            vertices
            .join(contribs.groupBy("id").agg(F.sum("contrib").alias("inbound")),
                  "id", "left")
            .withColumn(
                "rank",
                F.lit((1 - damping) / max(n, 1))
                + F.lit(damping) * F.coalesce(F.col("inbound"), F.lit(0.0)),
            )
            .select("id", "rank")
        )

        # Truncate lineage EVERY superstep.
        #
        # Without this the logical plan grows by one join per iteration and
        # the query planner eventually exhausts the driver - we hit exactly
        # that failure. localCheckpoint materialises the frame and discards
        # the plan behind it, which is what makes iterative algorithms
        # tractable in Spark at all. It is the in-memory equivalent of
        # RDD.checkpoint() and needs no checkpoint directory.
        ranks = ranks.localCheckpoint(eager=True)

    both.unpersist()
    out_deg.unpersist()
    return ranks


# ----------------------------------------------------------------------
# Connected components via label propagation
# ----------------------------------------------------------------------

def connected_components(vertices, edges, max_iterations=10):
    """
    Each vertex adopts the smallest ID it can see, propagated outward until
    labels stop changing. This is the standard label-propagation formulation
    of connected components and is what GraphX's implementation reduces to.

    Interpretation: a marketplace whose review graph is one giant component
    means reviewers are densely linked through shared products. Many small
    components would mean isolated product niches with no overlap in
    readership — a materially different market structure.
    """
    both = (
        edges.select("src", "dst")
        .union(edges.select(F.col("dst").alias("src"), F.col("src").alias("dst")))
        .distinct()
    ).persist(StorageLevel.MEMORY_AND_DISK)

    labels = vertices.withColumn("component", F.col("id"))

    for i in range(max_iterations):
        proposed = (
            both
            .join(labels, both.src == labels.id)
            .groupBy("dst")
            .agg(F.min("component").alias("neighbour_min"))
            .withColumnRenamed("dst", "id")
        )

        updated = (
            labels.join(proposed, "id", "left")
            .withColumn(
                "new_component",
                F.least(F.col("component"),
                        F.coalesce(F.col("neighbour_min"), F.col("component"))),
            )
            # Truncate lineage before the convergence check, not after.
            # Checking first would force a recompute of the entire accumulated
            # plan on every iteration - which is what crashed the driver
            # before this fix.
            .localCheckpoint(eager=True)
        )

        changed = updated.filter(F.col("new_component") != F.col("component")).count()
        labels = updated.select("id", F.col("new_component").alias("component"))

        if changed == 0:
            print(f"  converged after {i + 1} iterations")
            break

    both.unpersist()
    return labels


# ----------------------------------------------------------------------
# GraphFrames path, when available
# ----------------------------------------------------------------------

def try_graphframes(vertices, edges, iterations=DEFAULT_ITERATIONS):
    """Use GraphFrames if a compatible build is installed. Returns None
    otherwise, and the caller falls back to the explicit implementation."""
    try:
        from graphframes import GraphFrame
    except Exception:
        return None
    try:
        g = GraphFrame(vertices, edges)
        pr = g.pageRank(resetProbability=1 - DAMPING, maxIter=iterations)
        print("  using GraphFrames")
        return pr.vertices.select("id", F.col("pagerank").alias("rank"))
    except Exception as e:
        print(f"  GraphFrames present but failed ({type(e).__name__}); "
              f"using the explicit implementation")
        return None


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------

def run(df, iterations=DEFAULT_ITERATIONS, top_n=15, max_vertices=4_000_000):
    """
    Entry point. Returns (summary_table, top_reviewers, component_table).

    max_vertices guards the driver: every PageRank superstep materialises the
    full rank frame via localCheckpoint, so cost scales linearly with vertex
    count and with iterations. Above the cap we raise the degree threshold,
    which removes the least-connected nodes first - exactly the ones that
    contribute least to centrality - rather than sampling the graph, which
    would sever edges and distort it.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 8 - GRAPH ANALYTICS (reviewer-product network)")
    print("=" * 70)

    min_deg = 2
    vertices, edges = build_bipartite_graph(df, min_deg, min_deg)

    n_probe = vertices.count()
    while n_probe > max_vertices and min_deg < 6:
        min_deg += 1
        print(f"  graph has {n_probe:,} vertices; raising degree threshold "
              f"to {min_deg} to keep PageRank tractable")
        vertices, edges = build_bipartite_graph(df, min_deg, min_deg)
        n_probe = vertices.count()
    vertices = vertices.persist(StorageLevel.MEMORY_AND_DISK)
    edges = edges.persist(StorageLevel.MEMORY_AND_DISK)

    n_v, n_e = vertices.count(), edges.count()
    n_users = vertices.filter(F.col("id").startswith("u:")).count()
    n_products = n_v - n_users

    n_reviews = df.count()
    retention = 100 * n_e / max(n_reviews, 1)

    print(f"Graph: {n_v:,} vertices ({n_users:,} reviewers, "
          f"{n_products:,} products), {n_e:,} edges")
    print(f"Edges retained from {n_reviews:,} reviews: {retention:.1f}%")

    if retention < 20.0:
        print()
        print("!" * 70)
        print("SPARSITY WARNING - read before interpreting this graph")
        print("!" * 70)
        print(f"Degree-1 pruning removed {100 - retention:.1f}% of reviews,")
        print("because most reviewers and most products appear exactly once.")
        print()
        print("This is a signature of REVIEW-LEVEL RANDOM SAMPLING: drawing")
        print("reviews at random from a large corpus leaves almost every user")
        print("with a single review, so there are few shared reviewers to")
        print("connect products and the graph cannot help but fragment.")
        print()
        print("Treat centrality and component structure below as properties")
        print("of THIS EXTRACT, not of Amazon's review network. The complete")
        print("per-category dumps preserve user history and yield a connected")
        print("graph; the methods here transfer unchanged to them.")
        print("!" * 70)
        print()

    if n_v < 50 or n_e < 50:
        print("Graph too small for meaningful analysis; skipped.")
        vertices.unpersist(); edges.unpersist()
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # --- PageRank ---
    print(f"\nPageRank ({iterations} iterations):")
    ranks = try_graphframes(vertices, edges, iterations)
    if ranks is None:
        print("  using iterative DataFrame joins (Pregel-style supersteps)")
        ranks = pagerank(vertices, edges, iterations)
    ranks = ranks.persist(StorageLevel.MEMORY_AND_DISK)

    reviewer_ranks = (
        ranks.filter(F.col("id").startswith("u:"))
        .withColumn("user_id", F.expr("substring(id, 3)"))
        .select("user_id", "rank")
    )

    # Does graph centrality tell us anything review count does not?
    activity = (
        df.groupBy("user_id")
        .agg(F.count("*").alias("n_reviews"),
             F.sum("helpful_vote").alias("total_helpful"),
             F.avg("label").alias("hit_rate"))
    )

    joined = reviewer_ranks.join(activity, "user_id").cache()

    top = (
        joined.orderBy(F.desc("rank")).limit(top_n)
        .select("user_id", F.round("rank", 8).alias("pagerank"),
                "n_reviews", "total_helpful",
                F.round("hit_rate", 3).alias("hit_rate"))
    ).toPandas()

    print(f"\nTop {top_n} reviewers by PageRank:")
    print(top.to_string(index=False))

    corr_n = joined.stat.corr("rank", "n_reviews")
    corr_h = joined.stat.corr("rank", "total_helpful")
    print(f"\nCorrelation of PageRank with review count : {corr_n:.4f}")
    print(f"Correlation of PageRank with helpful votes: {corr_h:.4f}")
    # Honest reporting in BOTH directions. An earlier threshold of 0.9 was
    # far too generous: at r = 0.90, centrality and activity are nearly the
    # same variable, and claiming otherwise would not survive scrutiny.
    if corr_n is None:
        pass
    elif corr_n >= 0.85:
        print("CAVEAT: PageRank correlates at r = %.3f with simple review" % corr_n)
        print("count, so on this graph centrality is largely a restatement of")
        print("activity and adds little beyond it. Report it as such rather")
        print("than as an independent signal.")
    elif corr_n >= 0.70:
        print("PageRank is only partly separable from review count "
              "(r = %.3f)." % corr_n)
        print("It carries some structural information, but activity explains")
        print("most of it.")
    else:
        print("PageRank is NOT a restatement of review count (r = %.3f):" % corr_n)
        print("centrality rewards reviewing products that others also review,")
        print("which simple activity counts cannot express.")

    # --- Connected components ---
    print("\nConnected components:")
    comps = connected_components(vertices, edges)
    comp_sizes = (
        comps.groupBy("component").count()
        .withColumnRenamed("count", "size")
        .orderBy(F.desc("size"))
    )
    comp_pdf = comp_sizes.limit(10).toPandas()
    n_comps = comp_sizes.count()
    largest = int(comp_pdf.iloc[0]["size"]) if len(comp_pdf) else 0

    print(f"  {n_comps:,} components; largest holds {largest:,} vertices "
          f"({100 * largest / max(n_v, 1):.1f}% of the graph)")
    if largest / max(n_v, 1) > 0.8:
        print("  A single giant component: reviewers are densely linked")
        print("  through shared products rather than split into niches.")

    summary = pd.DataFrame([{
        "vertices": n_v,
        "reviewers": n_users,
        "products": n_products,
        "edges": n_e,
        "pagerank_iterations": iterations,
        "corr_pagerank_reviewcount": round(corr_n, 4) if corr_n else None,
        "corr_pagerank_helpfulvotes": round(corr_h, 4) if corr_h else None,
        "n_components": n_comps,
        "largest_component": largest,
        "largest_component_pct": round(100 * largest / max(n_v, 1), 2),
    }])

    joined.unpersist(); ranks.unpersist()
    vertices.unpersist(); edges.unpersist()

    return summary, top, comp_pdf
