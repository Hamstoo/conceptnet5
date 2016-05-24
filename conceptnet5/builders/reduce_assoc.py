from collections import defaultdict
import argparse
from conceptnet5.uri import split_uri, join_uri


def concept_is_bad(uri):
    """
    Skip concepts that are unlikely to be useful.

    A concept containing too many underscores is probably a long, overly
    specific phrase, possibly mis-parsed. A concept with a colon is probably
    detritus from a wiki.
    """
    return ':' in uri or uri.count('_') >= 3 or uri.startswith('/a/') or uri.endswith('/neg')


def generalized_uris(uri):
    pieces = split_uri(uri)
    if len(pieces) >= 5:
        return [uri, join_uri(*pieces[:3])]
    else:
        return [join_uri(*pieces[:3])]


def reduce_assoc(filename, output_filename, cutoff=4, en_cutoff=4, verbose=True):
    """
    Removes uncommon associations and associations unlikely to be useful.
    This function expects files of the form part_*.csv in `dirname` and will
    create `reduced.csv` in `dirname`.

    All concepts that occur fewer than `cutoff` times will be removed.
    All English concepts that occur fewer than `en_cutoff` times will be removed.
    """
    counts = defaultdict(int)
    with open(filename, encoding='utf-8') as file:
        for line in file:
            left, right, *_ = line.rstrip().split('\t')
            for gleft in generalized_uris(left):
                for gright in generalized_uris(right):
                    counts[gleft] += 1
                    counts[gright] += 1

    filtered_concepts = {
        concept for (concept, count) in counts.items()
        if (
            count >= en_cutoff or
            (not concept.startswith('/c/en/') and count >= cutoff)
        )
    }

    with open(output_filename, 'w', encoding='utf-8') as out:
        with open(filename, encoding='utf-8') as file:
            for line in file:
                left, right, value, dataset, rel = line.rstrip().split('\t', 4)
                if concept_is_bad(left) or concept_is_bad(right):
                    continue
                fvalue = float(value)
                for gleft in generalized_uris(left):
                    for gright in generalized_uris(right):
                        if (
                            gleft in filtered_concepts and
                            gright in filtered_concepts and
                            fvalue != 0
                        ):
                            line = '\t'.join([gleft, gright, value, dataset, rel])
                            print(line, file=out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_filename')
    parser.add_argument('output_filename')

    args = parser.parse_args()
    reduce_assoc(args.input_filename, args.output_filename)

