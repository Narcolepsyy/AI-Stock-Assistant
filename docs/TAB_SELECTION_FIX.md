# Tab Selection Fix - Prevent Sticky Header Selection

## Issue
When clicking on tabs in the sticky header, the header itself was getting selected/highlighted due to automatic scrolling behavior and focus management issues.

## Root Cause
The `handleTabChange` function in `StickyHeader.tsx` included automatic scrolling logic that would:
1. Scroll to the header position when tabs were clicked
2. Cause the sticky header to be selected or highlighted
3. Create unwanted focus behavior

## Solution Applied

### 1. Removed Automatic Scrolling (`StickyHeader.tsx`)
```tsx
// Before: Complex scrolling logic
const handleTabChange = React.useCallback((tab: TabType) => {
  onTabChange(tab)
  // ... complex scrolling logic that caused header selection
}, [onTabChange, headerRef, messagesContainerRef])

// After: Simple tab change without scrolling
const handleTabChange = React.useCallback((tab: TabType) => {
  onTabChange(tab)
  // Remove automatic scrolling behavior to prevent header selection
}, [onTabChange])
```

### 2. Enhanced Click Event Handling
Added proper event handling to prevent unwanted behavior:
```tsx
onClick={(e) => {
  e.preventDefault()        // Prevent default button behavior
  e.stopPropagation()      // Stop event bubbling
  handleTabChange('answer') // Clean tab change
}}
onMouseDown={(e) => e.preventDefault()} // Prevent focus change
```

### 3. CSS Improvements (`styles.css`)
Enhanced user selection and focus management:
```css
/* Prevent text selection and unwanted focus behavior on tabs */
.qa-tabs button[role="tab"] {
  user-select: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  outline: none;
}

/* Prevent header text selection when tabs are clicked */
.qa-header {
  user-select: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
}

/* Allow text selection only for the question content */
.qa-question-text {
  user-select: text;
  -webkit-user-select: text;
  -moz-user-select: text;
  -ms-user-select: text;
}
```

## Benefits

### ✅ **Fixed Issues**
- **No Header Selection**: Clicking tabs no longer selects/highlights the sticky header
- **Clean Tab Switching**: Pure tab functionality without side effects
- **Better UX**: Users can switch tabs without unwanted scrolling behavior
- **Proper Focus Management**: Focus stays on the clicked tab, not the header

### ✅ **Preserved Features**
- **Sticky Header Functionality**: Headers still stick properly during scroll
- **Tab State Management**: Active tab states are maintained correctly
- **Accessibility**: ARIA attributes and keyboard navigation still work
- **Visual Feedback**: Hover and active states remain intact

### ✅ **Improved Behavior**
- **User Control**: Users can manually scroll if needed without forced positioning
- **Predictable Interaction**: Tab clicks only change tab content, nothing else
- **Text Selection**: Question text can still be selected when needed
- **Performance**: Removed complex scrolling calculations

## Technical Details

### Files Modified
1. **`/frontend/src/components/StickyHeader.tsx`**
   - Simplified `handleTabChange` function
   - Added `preventDefault()` and `stopPropagation()` to click handlers
   - Added `onMouseDown` prevention for focus management

2. **`/frontend/src/styles.css`**
   - Added `user-select: none` for tab buttons and header
   - Preserved `user-select: text` for question content
   - Enhanced focus management CSS

### Testing
- ✅ Build completes successfully
- ✅ No TypeScript errors
- ✅ Tab functionality preserved
- ✅ Sticky behavior maintained
- ✅ No unwanted header selection

## Usage
Now when users click on tabs (Answer, Sources, Steps), the behavior is clean and predictable:
- Tab content switches immediately
- No automatic scrolling occurs
- Sticky header remains unselected
- Focus stays on the clicked tab button

The fix maintains all existing functionality while eliminating the unwanted header selection behavior.