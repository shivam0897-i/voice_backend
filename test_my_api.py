"""
Official evaluation script from the hackathon guide, configured with our 5 test files.
This mirrors EXACTLY what the evaluator will run.
"""
import requests
import base64
import json

def evaluate_voice_detection_api(endpoint_url, api_key, test_files):
    if not endpoint_url:
        print("Error: Endpoint URL is required")
        return False
    if not test_files or len(test_files) == 0:
        print("Error: No test files provided")
        return False

    total_files = len(test_files)
    score_per_file = 100 / total_files
    total_score = 0
    file_results = []

    print(f"\n{'='*60}")
    print(f"Starting Evaluation")
    print(f"{'='*60}")
    print(f"Endpoint: {endpoint_url}")
    print(f"Total Test Files: {total_files}")
    print(f"Score per File: {score_per_file:.2f}")
    print(f"{'='*60}\n")

    for idx, file_data in enumerate(test_files):
        language = file_data.get('language', 'English')
        file_path = file_data.get('file_path', '')
        expected_classification = file_data.get('expected_classification', '')

        print(f"Test {idx + 1}/{total_files}: {file_path}")

        if not file_path or not expected_classification:
            file_results.append({'fileIndex': idx, 'status': 'skipped', 'score': 0})
            print(f"   Skipped: Missing file path or expected classification\n")
            continue

        try:
            with open(file_path, 'rb') as audio_file:
                audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')
        except Exception as e:
            file_results.append({'fileIndex': idx, 'status': 'failed', 'message': f'Failed to read: {e}', 'score': 0})
            print(f"   Failed to read file: {e}\n")
            continue

        headers = {'Content-Type': 'application/json', 'x-api-key': api_key}
        request_body = {'language': language, 'audioFormat': 'mp3', 'audioBase64': audio_base64}

        try:
            response = requests.post(endpoint_url, headers=headers, json=request_body, timeout=30)

            if response.status_code != 200:
                file_results.append({'fileIndex': idx, 'status': 'failed', 'message': f'HTTP {response.status_code}', 'score': 0})
                print(f"   HTTP Status: {response.status_code}")
                print(f"   Response: {response.text[:200]}\n")
                continue

            response_data = response.json()

            if not isinstance(response_data, dict):
                file_results.append({'fileIndex': idx, 'status': 'failed', 'message': 'Not a JSON object', 'score': 0})
                print(f"   Invalid response type\n")
                continue

            response_status = response_data.get('status', '')
            response_classification = response_data.get('classification', '')
            confidence_score = response_data.get('confidenceScore', None)

            if not response_status or not response_classification or confidence_score is None:
                file_results.append({'fileIndex': idx, 'status': 'failed', 'message': 'Missing required fields', 'score': 0})
                print(f"   Missing required fields")
                print(f"   Response: {json.dumps(response_data, indent=2)[:200]}\n")
                continue

            if response_status != 'success':
                file_results.append({'fileIndex': idx, 'status': 'failed', 'message': f'Status: {response_status}', 'score': 0})
                print(f"   Status not 'success': {response_status}\n")
                continue

            if not isinstance(confidence_score, (int, float)) or confidence_score < 0 or confidence_score > 1:
                file_results.append({'fileIndex': idx, 'status': 'failed', 'message': f'Invalid confidence: {confidence_score}', 'score': 0})
                print(f"   Invalid confidence score: {confidence_score}\n")
                continue

            valid_classifications = ['HUMAN', 'AI_GENERATED']
            if response_classification not in valid_classifications:
                file_results.append({'fileIndex': idx, 'status': 'failed', 'message': f'Invalid classification: {response_classification}', 'score': 0})
                print(f"   Invalid classification: {response_classification}\n")
                continue

            # Score calculation
            file_score = 0
            if response_classification == expected_classification:
                if confidence_score >= 0.8:
                    file_score = score_per_file
                    confidence_tier = "100%"
                elif confidence_score >= 0.6:
                    file_score = score_per_file * 0.75
                    confidence_tier = "75%"
                elif confidence_score >= 0.4:
                    file_score = score_per_file * 0.5
                    confidence_tier = "50%"
                else:
                    file_score = score_per_file * 0.25
                    confidence_tier = "25%"
                total_score += file_score
                file_results.append({'fileIndex': idx, 'status': 'success', 'matched': True, 'score': round(file_score, 2),
                                     'actualClassification': response_classification, 'confidenceScore': confidence_score})
                print(f"   CORRECT: {response_classification}")
                print(f"   Confidence: {confidence_score:.2f} -> {confidence_tier} of points")
                print(f"   Score: {file_score:.2f}/{score_per_file:.2f}\n")
            else:
                file_results.append({'fileIndex': idx, 'status': 'success', 'matched': False, 'score': 0,
                                     'actualClassification': response_classification, 'confidenceScore': confidence_score})
                print(f"   WRONG: {response_classification} (Expected: {expected_classification})")
                print(f"   Score: 0/{score_per_file:.2f}\n")

        except requests.exceptions.Timeout:
            file_results.append({'fileIndex': idx, 'status': 'failed', 'message': 'Timeout (>30s)', 'score': 0})
            print(f"   TIMEOUT: Request took longer than 30 seconds\n")
        except requests.exceptions.ConnectionError:
            file_results.append({'fileIndex': idx, 'status': 'failed', 'message': 'Connection error', 'score': 0})
            print(f"   CONNECTION ERROR\n")
        except Exception as e:
            file_results.append({'fileIndex': idx, 'status': 'failed', 'message': str(e), 'score': 0})
            print(f"   ERROR: {e}\n")

    final_score = round(total_score)

    print(f"{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total Files Tested: {total_files}")
    print(f"Final Score: {final_score}/100")
    print(f"{'='*60}\n")

    successful = sum(1 for r in file_results if r.get('matched', False))
    failed = sum(1 for r in file_results if r['status'] == 'failed')
    wrong = sum(1 for r in file_results if r['status'] == 'success' and not r.get('matched', False))

    print(f"Correct Classifications: {successful}/{total_files}")
    print(f"Wrong Classifications: {wrong}/{total_files}")
    print(f"Failed/Errors: {failed}/{total_files}\n")

    with open('evaluation_results.json', 'w') as f:
        json.dump({'finalScore': final_score, 'totalFiles': total_files, 'scorePerFile': round(score_per_file, 2),
                   'successfulClassifications': successful, 'wrongClassifications': wrong, 'failedTests': failed,
                   'fileResults': file_results}, f, indent=2)
    print(f"Detailed results saved to: evaluation_results.json\n")
    return True


if __name__ == '__main__':
    ENDPOINT_URL = 'https://shivam-2211-voice-detection-api.hf.space/api/voice-detection'
    API_KEY = 'sk_test_voice_detection_2026'

    DIR = r'c:\Users\shiva\OneDrive\Desktop\Voice Project\voice-detection-api\drive-download-20260216T053632Z-1-001'

    TEST_FILES = [
        {'language': 'English', 'file_path': f'{DIR}\\English_voice_AI_GENERATED.mp3', 'expected_classification': 'AI_GENERATED'},
        {'language': 'Hindi',   'file_path': f'{DIR}\\Hindi_Voice_HUMAN.mp3',          'expected_classification': 'HUMAN'},
        {'language': 'Malayalam','file_path': f'{DIR}\\Malayalam_AI_GENERATED.mp3',     'expected_classification': 'AI_GENERATED'},
        {'language': 'Tamil',   'file_path': f'{DIR}\\TAMIL_VOICE__HUMAN.mp3',         'expected_classification': 'HUMAN'},
        {'language': 'Telugu',  'file_path': f'{DIR}\\Telugu_Voice_AI_GENERATED.mp3',  'expected_classification': 'AI_GENERATED'},
    ]

    evaluate_voice_detection_api(ENDPOINT_URL, API_KEY, TEST_FILES)
