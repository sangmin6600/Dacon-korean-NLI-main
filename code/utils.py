def read_text(fn):
    with open(fn, 'r') as f:
        lines = f.readlines()
        premises, hypothesises, labels = [], [], []
        for line in lines[1:]:
            if line.strip() != '':
            # The file should have tab delimited two columns.
            # First column indicates label field,
            # and second column indicates text field.
                index, premise, hypothesis, label = line.strip().split('\t')
                premises += [premise]
                hypothesises += [hypothesis]
                labels += [label]

    return premises, hypothesises, labels


def get_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    total_norm = 0

    try:
        for p in parameters:
            total_norm += (p.grad.data**norm_type).sum()
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm


def get_parameter_norm(parameters, norm_type=2):
    total_norm = 0

    try:
        for p in parameters:
            total_norm += (p.data**norm_type).sum()
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm
