# scripts/reset_refund_state.py
"""
P1 修复配套: 重置退款相关数据,让用户能在干净状态下复测。

用法:
    python scripts/reset_refund_state.py --username alice
    python scripts/reset_refund_state.py --user-id 2
    python scripts/reset_refund_state.py --username alice --wipe-refunds
    python scripts/reset_refund_state.py --username alice --dry-run

行为:
  默认 --reset-memory
    把该用户的 ConversationSession.working_memory_json 重置为默认值,
    关键: 清掉 collected_slots.refund_submitted_id / user_confirmed /
    refund_reason / workflow_stage / last_tool_result,避免退款 FSM 上一笔
    的终态被新请求继承后短路到 DONE。

  --wipe-refunds (可选,需二次确认)
    删除该用户所有 RefundApplication 与 AuditLog。仅在你想完全从零开始
    测退款全链路时使用(会丢失审核/申请数据)。

  --dry-run
    只打印会做的动作,不实际改库。

注意: 重置 ConversationSession 后,前端 BrowserState 里缓存的 conversation_id
仍然指向旧 session。建议同时在前端页面 Ctrl+Shift+R 强制刷新或切到无痕窗口,
让 BrowserState 重新生成,再触发新一轮请求。
"""
import argparse
import asyncio
import os
import sys

sys.path.append(os.getcwd())

from sqlmodel import select, delete

from app.conversation.state_manager import default_working_memory
from app.core.database import async_session_maker
from app.models.audit import AuditLog
from app.models.conversation import ConversationSession
from app.models.refund import RefundApplication
from app.models.user import User


async def resolve_user(session, *, username: str | None, user_id: int | None) -> User | None:
    if user_id is not None:
        return await session.get(User, user_id)
    if username:
        result = await session.exec(select(User).where(User.username == username))
        return result.first()
    return None


async def reset_working_memory(session, user: User, *, dry_run: bool) -> int:
    stmt = select(ConversationSession).where(ConversationSession.user_id == user.id)
    sessions = list((await session.exec(stmt)).all())
    if not sessions:
        print(f"  - {user.username}: 无 ConversationSession,跳过")
        return 0
    for s in sessions:
        print(
            f"  - session={s.conversation_id} 旧 stage={s.working_memory_json.get('workflow_stage')} "
            f"submitted_id={(s.working_memory_json.get('collected_slots') or {}).get('refund_submitted_id')}"
        )
    if dry_run:
        print(f"  [dry-run] 将重置 {len(sessions)} 个 session 的 working_memory_json")
        return len(sessions)
    for s in sessions:
        s.working_memory_json = default_working_memory()
        s.conversation_summary = None
        s.active_domain = None
        session.add(s)
    await session.commit()
    print(f"  ✓ 已重置 {len(sessions)} 个 session")
    return len(sessions)


async def wipe_refunds(session, user: User, *, dry_run: bool) -> tuple[int, int]:
    refund_stmt = delete(RefundApplication).where(RefundApplication.user_id == user.id)
    audit_stmt = delete(AuditLog).where(AuditLog.user_id == user.id)
    if dry_run:
        # dry-run 模式下不实际 delete,只统计数量
        from sqlmodel import func
        r_count = (await session.exec(
            select(func.count()).select_from(RefundApplication).where(RefundApplication.user_id == user.id)
        )).one()
        a_count = (await session.exec(
            select(func.count()).select_from(AuditLog).where(AuditLog.user_id == user.id)
        )).one()
        print(f"  [dry-run] 将删除 {r_count} 条 RefundApplication 和 {a_count} 条 AuditLog")
        return r_count, a_count
    r = await session.exec(refund_stmt)
    a = await session.exec(audit_stmt)
    await session.commit()
    print(f"  ✓ 已删除 RefundApplication={r.rowcount} 条, AuditLog={a.rowcount} 条")
    return r.rowcount or 0, a.rowcount or 0


async def main():
    parser = argparse.ArgumentParser(description="重置退款相关数据(配合 P1 修复)")
    parser.add_argument("--username", help="按用户名定位")
    parser.add_argument("--user-id", type=int, help="按 user_id 定位")
    parser.add_argument(
        "--wipe-refunds",
        action="store_true",
        help="同时删除该用户的 RefundApplication 和 AuditLog(谨慎,需二次确认)",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印动作,不实际改库")
    parser.add_argument("--yes", action="store_true", help="跳过二次确认(自动化场景)")
    args = parser.parse_args()

    if not args.username and not args.user_id:
        parser.error("需要 --username 或 --user-id")

    if args.wipe_refunds and not args.yes:
        confirm = input(
            "⚠️  --wipe-refunds 会删除该用户全部退款申请与审核日志,确认? (yes/no): "
        ).strip().lower()
        if confirm != "yes":
            print("已取消")
            return

    async with async_session_maker() as session:
        user = await resolve_user(session, username=args.username, user_id=args.user_id)
        if not user:
            print(f"❌ 未找到用户 (username={args.username}, user_id={args.user_id})")
            return
        print(f"目标用户: id={user.id} username={user.username}")

        print("\n[1/2] 重置 working memory ...")
        await reset_working_memory(session, user, dry_run=args.dry_run)

        if args.wipe_refunds:
            print("\n[2/2] 清除退款申请与审核日志 ...")
            await wipe_refunds(session, user, dry_run=args.dry_run)
        else:
            print("\n[2/2] 跳过 (未传 --wipe-refunds; 默认保留 RefundApplication/AuditLog)")

    print(
        "\n下一步: 在前端页面 Ctrl+Shift+R 强制刷新(或切无痕窗口),让 BrowserState "
        "重新生成,然后重新发起退款测试。"
    )


if __name__ == "__main__":
    asyncio.run(main())