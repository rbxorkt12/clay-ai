import { Sandbox } from '@e2b/code-interpreter'
import { NextResponse } from 'next/server'

export const runtime = 'edge'

export async function POST(req: Request) {
  try {
    const { fragment, userID, apiKey } = await req.json()

    if (!fragment || !userID) {
      return NextResponse.json(
        { error: 'Missing required parameters' },
        { status: 400 },
      )
    }

    // Create a new sandbox instance with the latest E2B features
    const sandbox = await Sandbox.create(fragment.template || 'base', {
      apiKey: process.env.E2B_API_KEY,
      metadata: { userID },
      timeoutMs: 10 * 60 * 1000, // 10 minutes
    })

    // Initialize the sandbox with the fragment code
    if (fragment.code) {
      await sandbox.files.write('/home/user/code.py', fragment.code)
      const result = await sandbox.commands.run('python /home/user/code.py')

      return NextResponse.json({
        output: result.stdout,
        error: result.stderr,
        exitCode: result.exitCode,
      })
    }

    return NextResponse.json({
      error: 'No code provided in fragment',
      status: 400,
    })
  } catch (error) {
    console.error('Sandbox error:', error)
    return NextResponse.json(
      { error: 'Failed to create sandbox' },
      { status: 500 },
    )
  }
}
