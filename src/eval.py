import string
import re
import json

try:
    import inflect
    ENGINE = inflect.engine()
except ImportError:
    ENGINE = None

def acc_score(prediction, ground_truth):
    """Match when the normalized prediction contains the normalized gold answer."""
    pred = normalize_answer(prediction)
    gold = normalize_answer(ground_truth)
    # Empty predictions/answers should not match (even if both empty)
    if not pred or not gold:
        return 0
    return 1 if gold in pred else 0

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Compute metric over all ground truths and return the maximum.
    
    Note: metric_fn receives already-normalized strings, so it should NOT
    re-normalize them. This avoids double-normalization.
    """
    scores_for_ground_truths = []
    if not isinstance(ground_truths, list):
        ground_truths = [ground_truths]
    # Normalize prediction once, outside the loop
    norm_prediction = normalize_answer(prediction)
    for ground_truth in ground_truths:
        norm_ground_truth = normalize_answer(ground_truth)
        score = metric_fn(norm_prediction, norm_ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths) if scores_for_ground_truths else 0

def normalize_answer(s, p=ENGINE):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')
    
    def convert_numbers_to_words(text):
        if p is None:
            return text
        words = text.split()
        result = []
        for word in words:
            # ``str.isdigit()`` is True for Unicode digit look-alikes such as
            # superscripts ("²") and circled digits, which ``int()`` rejects.
            # Restrict the conversion to plain ASCII decimals.
            if word.isascii() and word.isdecimal() and int(word) < 100:
                word_in_words = p.number_to_words(int(word))
                result.append(word_in_words)
            else:
                result.append(word)
        return ' '.join(result)

    return white_space_fix(remove_articles(handle_punc(convert_numbers_to_words(lower(replace_underscore(s)))))).strip()


def get_ground_truths(answer):
    answers = answer if isinstance(answer, list) else [answer]
    return [normalize_answer(item) for item in answers]


def extract_answer(text):
    pattern = r'"Answer":\s*"([^"]+)"'
    matches = re.findall(pattern, text)
    return matches[-1] if matches else text


def evaluate(dataset, predictions, cot_format=False, detailed_output=False):
    prediction_details = []
    total_acc = 0
    num_items = 0
    
    for item in dataset:
        prediction = predictions.get(item['id'], "") if 'id' in item else ""
        
        if cot_format:
            prediction = extract_answer(prediction)
            
        ground_truth = item['answer']
        
        acc_score_val = metric_max_over_ground_truths(
            acc_score, prediction, ground_truth)
        
        total_acc += acc_score_val
        num_items += 1
        
        if detailed_output:
            prediction_details.append({
                "id": item.get('id', ''),
                "question": item['question'],
                "answer": ground_truth,
                "prediction": prediction,
                "acc": acc_score_val
            })
    
    avg_acc = 100.0 * total_acc / num_items if num_items > 0 else 0
    
    result = {
        "num_items": num_items,
        "acc": avg_acc
    }
    
    if detailed_output:
        result["details"] = prediction_details
        
    return result

def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description='Evaluate saved CSRAG predictions.')
    parser.add_argument('--data', required=True)
    parser.add_argument('--predictions', required=True)
    args = parser.parse_args()
    samples = json.loads(Path(args.data).read_text(encoding='utf-8'))
    saved = json.loads(Path(args.predictions).read_text(encoding='utf-8'))
    records = saved['results'] if isinstance(saved, dict) else saved
    predictions = {str(row['id']): row['final_response'] for row in records}
    expected = {str(row['id']) for row in samples}
    if len(predictions) != len(records) or set(predictions) != expected:
        raise ValueError('Prediction IDs must be unique and match the dataset exactly.')
    rows = [{**row, 'id': str(row['id'])} for row in samples]
    print(json.dumps(evaluate(rows, predictions, cot_format=True), indent=2))


if __name__ == '__main__':
    main()
