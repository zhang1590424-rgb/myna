import AppKit
import Foundation

private let appName = "Myna"
private let host = "127.0.0.1"
private let port = 4180
private let healthURL = URL(string: "http://\(host):\(port)/api/health")!
private let appURL = URL(string: "http://\(host):\(port)")!

private func projectRootURL() -> URL {
    Bundle.main.bundleURL.deletingLastPathComponent()
}

private func showAlert(title: String, message: String) {
    DispatchQueue.main.async {
        NSApp.activate(ignoringOtherApps: true)
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = message
        alert.alertStyle = .warning
        alert.addButton(withTitle: "好")
        alert.runModal()
    }
}

private func checkHealth(timeout: TimeInterval = 1.2, completion: @escaping (Bool) -> Void) {
    var request = URLRequest(url: healthURL)
    request.timeoutInterval = timeout
    URLSession.shared.dataTask(with: request) { _, response, _ in
        let ok = (response as? HTTPURLResponse)?.statusCode == 200
        completion(ok)
    }.resume()
}

private final class AppDelegate: NSObject, NSApplicationDelegate {
    private let rootURL = projectRootURL()
    private var serverProcess: Process?
    private var logHandle: FileHandle?
    private var readinessTimer: Timer?
    private var readinessAttempts = 0
    private var openedBrowser = false
    private var startedByThisApp = false

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        installMenu()
        startOrOpen()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false
    }

    func applicationWillTerminate(_ notification: Notification) {
        readinessTimer?.invalidate()
        if startedByThisApp, let process = serverProcess, process.isRunning {
            process.terminate()
        }
        try? logHandle?.close()
    }

    private func installMenu() {
        let mainMenu = NSMenu()
        let appMenuItem = NSMenuItem()
        mainMenu.addItem(appMenuItem)

        let appMenu = NSMenu(title: appName)
        let openItem = NSMenuItem(title: "打开 Myna", action: #selector(openMyna), keyEquivalent: "o")
        openItem.target = self
        appMenu.addItem(openItem)
        appMenu.addItem(.separator())
        appMenu.addItem(
            NSMenuItem(
                title: "退出 Myna",
                action: #selector(NSApplication.terminate(_:)),
                keyEquivalent: "q"
            )
        )
        appMenuItem.submenu = appMenu
        NSApp.mainMenu = mainMenu
    }

    private func startOrOpen() {
        checkHealth { [weak self] healthy in
            guard let self else { return }
            DispatchQueue.main.async {
                if healthy {
                    self.openMyna()
                    return
                }
                self.startServer()
            }
        }
    }

    private func startServer() {
        let pythonURL = rootURL.appendingPathComponent(".venv/bin/python")
        let startScriptURL = rootURL.appendingPathComponent("scripts/start.command")

        guard FileManager.default.isExecutableFile(atPath: pythonURL.path) else {
            let installHint = FileManager.default.fileExists(atPath: startScriptURL.path)
                ? "请先在项目目录运行 scripts/install.command 完成安装。旧入口「启动Myna.command」也仍可作为备用。"
                : "请确认 Myna.app 仍放在项目根目录，和 scripts、local_trainer、web 在同一层。"
            showAlert(title: "Myna 启动失败", message: "没有找到运行环境 .venv。\n\n\(installHint)")
            NSApp.terminate(nil)
            return
        }

        let runtimeURL = rootURL.appendingPathComponent("runtime", isDirectory: true)
        let logURL = runtimeURL.appendingPathComponent("myna-app.log")
        do {
            try FileManager.default.createDirectory(at: runtimeURL, withIntermediateDirectories: true)
            if !FileManager.default.fileExists(atPath: logURL.path) {
                FileManager.default.createFile(atPath: logURL.path, contents: nil)
            }
            logHandle = try FileHandle(forWritingTo: logURL)
            try logHandle?.seekToEnd()
        } catch {
            showAlert(title: "Myna 启动失败", message: "无法写入日志文件：\(error.localizedDescription)")
            NSApp.terminate(nil)
            return
        }

        let process = Process()
        process.executableURL = pythonURL
        process.currentDirectoryURL = rootURL
        process.arguments = [
            "-m",
            "uvicorn",
            "local_trainer.main:app",
            "--host",
            host,
            "--port",
            String(port),
        ]
        var environment = ProcessInfo.processInfo.environment
        environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        process.environment = environment
        process.standardOutput = logHandle
        process.standardError = logHandle
        process.terminationHandler = { [weak self] finishedProcess in
            DispatchQueue.main.async {
                guard let self else { return }
                self.readinessTimer?.invalidate()
                if !self.openedBrowser {
                    showAlert(
                        title: "Myna 启动失败",
                        message: "本地服务已退出（状态码 \(finishedProcess.terminationStatus)）。请查看 runtime/myna-app.log，或运行 scripts/doctor.command 检查环境。"
                    )
                }
            }
        }

        do {
            try process.run()
            serverProcess = process
            startedByThisApp = true
            waitUntilReady()
        } catch {
            showAlert(title: "Myna 启动失败", message: "无法启动本地服务：\(error.localizedDescription)")
            NSApp.terminate(nil)
        }
    }

    private func waitUntilReady() {
        readinessAttempts = 0
        readinessTimer?.invalidate()
        readinessTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] timer in
            guard let self else { return }
            self.readinessAttempts += 1
            checkHealth { healthy in
                DispatchQueue.main.async {
                    if healthy {
                        timer.invalidate()
                        self.openMyna()
                    } else if self.readinessAttempts >= 120 {
                        timer.invalidate()
                        showAlert(
                            title: "Myna 启动超时",
                            message: "本地服务 60 秒内没有就绪。请查看 runtime/myna-app.log，或运行 scripts/doctor.command 检查环境。"
                        )
                    }
                }
            }
        }
    }

    @objc private func openMyna() {
        openedBrowser = true
        NSWorkspace.shared.open(appURL)
    }
}

if ProcessInfo.processInfo.environment["MYNA_APP_DRY_RUN"] == "1" {
    let root = projectRootURL()
    print(root.path)
    print(root.appendingPathComponent(".venv/bin/python").path)
    print(root.appendingPathComponent("runtime/myna-app.log").path)
    exit(0)
}

let app = NSApplication.shared
private let delegate = AppDelegate()
app.delegate = delegate
app.run()
