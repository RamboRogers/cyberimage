"""
Tests for the Model Context Protocol (MCP) implementation
"""
import unittest
import json
import time
from flask import url_for
from app import create_app

class MCPTestCase(unittest.TestCase):
    """Tests for the MCP implementation"""

    def setUp(self):
        """Set up test app"""
        self.app = create_app({
            'TESTING': True,
            'ENABLE_RATE_LIMIT': False,  # Disable rate limiting for tests
        })
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()

    def tearDown(self):
        """Clean up after tests"""
        self.app_context.pop()

    def test_mcp_invalid_request(self):
        """Test that invalid requests are properly rejected"""
        # Test missing jsonrpc field
        response = self.client.post(
            '/api/mcp',
            json={'method': 'context.image_generation.models'}
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data['error']['code'], -32600)

        # Test invalid jsonrpc version
        response = self.client.post(
            '/api/mcp',
            json={'jsonrpc': '1.0', 'method': 'context.image_generation.models'}
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data['error']['code'], -32600)

    def test_mcp_method_not_found(self):
        """Test that an invalid method is properly rejected"""
        response = self.client.post(
            '/api/mcp',
            json={
                'jsonrpc': '2.0',
                'method': 'invalid.method',
                'id': '123'
            }
        )
        self.assertEqual(response.status_code, 404)
        data = json.loads(response.data)
        self.assertEqual(data['error']['code'], -32601)
        self.assertEqual(data['id'], '123')

    def test_mcp_models_method(self):
        """Test the models listing method"""
        response = self.client.post(
            '/api/mcp',
            json={
                'jsonrpc': '2.0',
                'method': 'context.image_generation.models',
                'id': '123'
            }
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['jsonrpc'], '2.0')
        self.assertEqual(data['id'], '123')
        self.assertIn('result', data)

        # Check that we have our models and the default model
        self.assertIn('models', data['result'])
        self.assertIn('default', data['result'])
        self.assertEqual(data['result']['default'], 'flux-2')

        # Check that we have at least one model
        self.assertTrue(len(data['result']['models']) > 0)

        # Check that flux-2 exists
        self.assertIn('flux-2', data['result']['models'])

    def test_mcp_generate_method_missing_params(self):
        """Test the generate method with missing parameters"""
        response = self.client.post(
            '/api/mcp',
            json={
                'jsonrpc': '2.0',
                'method': 'context.image_generation.generate',
                'params': {},
                'id': '123'
            }
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('error', data)

    def test_mcp_generate_method(self):
        """Test the generate method"""
        response = self.client.post(
            '/api/mcp',
            json={
                'jsonrpc': '2.0',
                'method': 'context.image_generation.generate',
                'params': {
                    'prompt': 'Test prompt for MCP',
                    'model': 'flux-2',
                    'settings': {
                        'num_images': 1
                    }
                },
                'id': '123'
            }
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)

        # Check that we have our job ID and status
        self.assertIn('result', data)
        self.assertIn('job_id', data['result'])
        self.assertIn('status', data['result'])
        self.assertEqual(data['result']['status'], 'pending')

        # Remember the job ID for the next test
        self.job_id = data['result']['job_id']

    def test_mcp_status_method_missing_params(self):
        """Test the status method with missing parameters"""
        response = self.client.post(
            '/api/mcp',
            json={
                'jsonrpc': '2.0',
                'method': 'context.image_generation.status',
                'params': {},
                'id': '123'
            }
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('error', data)

    def test_mcp_status_method_invalid_job(self):
        """Test the status method with an invalid job ID"""
        response = self.client.post(
            '/api/mcp',
            json={
                'jsonrpc': '2.0',
                'method': 'context.image_generation.status',
                'params': {
                    'job_id': 'invalid-job-id'
                },
                'id': '123'
            }
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('error', data)

    def test_mcp_workflow(self):
        """Test the complete MCP workflow"""
        # Skip in CI environment as we won't have GPU
        import os
        if os.environ.get('CI'):
            self.skipTest("Skipping complete workflow test in CI environment")

        # 1. Submit generation job
        response = self.client.post(
            '/api/mcp',
            json={
                'jsonrpc': '2.0',
                'method': 'context.image_generation.generate',
                'params': {
                    'prompt': 'Test complete workflow',
                    'settings': {
                        'num_images': 1,
                        'num_inference_steps': 2  # Use minimal steps for testing
                    }
                },
                'id': 'workflow-test'
            }
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        job_id = data['result']['job_id']

        # 2. Check status - should be pending or processing
        response = self.client.post(
            '/api/mcp',
            json={
                'jsonrpc': '2.0',
                'method': 'context.image_generation.status',
                'params': {
                    'job_id': job_id
                },
                'id': 'workflow-status'
            }
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('result', data)
        self.assertIn('status', data['result'])
        self.assertIn(data['result']['status'], ['pending', 'processing'])

if __name__ == '__main__':
    unittest.main()