import pandas as pd


def compute_output_validate(weeight1, weight2, bias, inputs, correct):
    outputs = []

    # Generate and check output
    for test_input, correct_output in zip(test_inputs, correct):
        linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias
        output = int(linear_combination >= 0)
        is_correct_string = 'Yes' if output == correct_output else 'No'
        outputs.append([test_input[0], test_input[1], linear_combination, output, is_correct_string])

    # Print output
    num_wrong = len([output[4] for output in outputs if output[4] == 'No'])
    output_frame = pd.DataFrame(outputs, columns=['Input 1', '  Input 2', '  Linear Combination', '  Activation Output', '  Is Correct'])
    if not num_wrong:
        print('Nice!  You got it all correct.\n')
    else:
        print('You got {} wrong.  Keep trying!\n'.format(num_wrong))
    print(output_frame.to_string(index=False))

# And
weight1 = 1
weight2 = 1
bias = -1.5

test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
correct_outputs = [False, False, False, True]

compute_output_validate(weight1, weight2, bias, test_inputs, correct_outputs)

# Or
weight1 = 1
weight2 = 1
bias = -0.5

correct_outputs = [False, True, True, True]

compute_output_validate(weight1, weight2, bias, test_inputs, correct_outputs)

# Not, work only on input 1
weight1 = -1
weight2 = 0
bias = 0

correct_outputs = [True, True, False, False]

compute_output_validate(weight1, weight2, bias, test_inputs, correct_outputs)
